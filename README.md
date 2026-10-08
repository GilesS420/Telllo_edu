# Tello EDU – autonoom pad volgen + plassen detecteren

```
tello_gui.py          ← start hier: grafische besturing (pad plannen, 3D, terrein, sensoren, camera)
tello_combined.py     ← live vliegen met het toetsenbord (ZQSD), handig om te testen/kalibreren
drone/
    app.py            kern: missies afvliegen, plassen detecteren, Jetson-koppeling
    config.py         alle instellingen (netwerk, snelheid, geofence, camera, detectie)
    navigator.py      positie + waypoints afvliegen (stap- of vloeiende modus)
    odometry.py       meet de echte beweging: camera (visuele odometrie), kompas, hoogte
    puddle_detector.py  plasdetectie (drempelmethode of getraind model) + samenvoegen
    jetson_link.py    UDP/JSON-communicatie met de Jetson
    telemetry.py      alle sensorwaarden over tijd + terreinkaart (barometer − ToF)
    sim.py            simulator: alles testen zonder drone (`--sim`), met dozen op de vloer
gui/                  onderdelen van de GUI: thema, instrumenten, 3D-weergave, kaart
    keypress.py       toetsenbord-hulp voor tello_combined.py
json_flights/         opgeslagen vluchten (JSON), te openen met Opslaan/Laden in de GUI
docs/                 afbeeldingen voor deze README
```

## Opbouw

```
 Jetson  ──(UDP 9000: mission / abort / ...)──▶  Laptop  ──(Wi-Fi, djitellopy)──▶  Tello EDU
 Jetson  ◀──(UDP 9001: puddle / status / ...)──  Laptop  ◀──(video onderste camera)──
```

De laptop hangt aan de Wi-Fi van de Tello. De Jetson moet de laptop via een **tweede netwerk**
kunnen bereiken (ethernetkabel of tweede Wi-Fi-adapter). Een alternatief: de Tello EDU in
*station mode* aan een router hangen (`ap <ssid> <wachtwoord>` sturen), dan zitten Tello,
laptop en Jetson op hetzelfde netwerk en maak je de drone aan met `Tello(host="<ip van de tello>")`.

## Coördinatenstelsel (“mission frame”)

Alles in **centimeter**:

* **x** = vooruit (richting van de neus van de drone bij de start)
* **y** = links
* **z** = hoogte boven de vloer
* **oorsprong** = waar de drone staat als het script start

De Jetson rekent echte-wereldcoördinaten om naar dit stelsel en de plascoördinaten die
terugkomen weer terug. Staat de drone op wereldpositie `(X0, Y0)` (meter) met zijn neus in
richting `a` (graden), dan is voor een wereldpunt `(X, Y, Z)`:

```
x = 100 * ( (X-X0)*cos(a) + (Y-Y0)*sin(a) )
y = 100 * (-(X-X0)*sin(a) + (Y-Y0)*cos(a) )
z = 100 * Z
```

In plaats daarvan kan de Jetson ook vóór het opstijgen `{"type":"set_pose","x":..,"y":..,"yaw":..}`
sturen en dan rechtstreeks in zijn eigen stelsel (in cm) werken.

## Installatie (in je venv)

```bash
pip install -r requirements.txt
# pas nodig als jullie model klaar is (kies één):
pip install ultralytics   # DETECTOR_BACKEND = "yolo"
pip install inference     # DETECTOR_BACKEND = "roboflow"
```

## Gebruik

```bash
python tello_gui.py --sim             # eerst proberen zonder drone
python tello_gui.py                   # echte drone (laptop op de Wi-Fi van de Tello)
python tello_gui.py --record dataset  # meteen camerabeelden opslaan voor Roboflow
python tello_combined.py              # live vliegen met het toetsenbord
```

![Tello GUI](docs/gui.png)

**Bovenbalk**: toestand, batterij, vliegtijd, hoogte, odometrie en of de Jetson berichten
stuurt. Meldingen (plas gevonden, waarschuwingen, fouten) verschijnen rechtsboven in plaats
van in pop-ups.

**Links: pad plannen en vliegen**
* **Waypoint**: vul x, y en z in (cm, pijltjes of scrollwiel = ±10) en druk op Enter of
  *Toevoegen*. Een waarde buiten de geofence kleurt meteen rood. Klik een punt in de lijst aan
  om het aan te passen (*Bijwerken*), te verplaatsen (▲▼) of te verwijderen (✕ of Delete).
  *Drone-positie* neemt de huidige positie over. **Ctrl+Z** maakt elke wijziging ongedaan.
* **Raster…** maakt een pad in banen (“grasmaaier”) over een rechthoek, om een gebied af te
  zoeken naar plassen en het terrein in kaart te brengen. De tussenafstand wordt voorgesteld
  op basis van wat de camera op die hoogte ziet.
* Boven de lijst staan het aantal punten, de lengte en een schatting van de vliegduur.
* **Opslaan/Laden** (Ctrl+S / Ctrl+O): vluchten als JSON in `json_flights/`
  (zie `json_flights/mission_example.json` voor het formaat).
* **Vliegen**: *Start missie*, *Ga naar punt* (blijft daarna hangen), *Opstijgen*,
  *Landen* (toets L) en **NOODSTOP** (toets X, motoren uit, de drone valt!).

**Midden: drie tabbladen** (Ctrl+1/2/3)
* **3D-weergave**: assenkruis bij de start (x rood = vooruit, y groen = links, z blauw = hoogte),
  oranje = gepland pad, blauw = actieve missie (ook van de Jetson), groen = bereikte punten,
  roze = gevlogen spoor, het drone-modelletje draait mee met de richting en kantelt met
  pitch/roll (rode rotors = voorkant), cyaan = plassen en wat de camera nu ziet, gekleurde
  tegels = gemeten terrein. Slepen = draaien, scrollen = zoomen, of kies *3D*, *Boven*,
  *Zijkant* of *Achter*. De assen schuiven vloeiend mee in plaats van te verspringen.
* **Kaart & terrein**: bovenaanzicht met de terreinkaart. **Klik** om een punt toe te voegen
  (hoogte = het z-veld), **sleep** een punt om het te verplaatsen, **rechtsklik** om het te
  wissen, scroll om te zoomen. Plassen staan op ware grootte. Rechts de terreinanalyse,
  onderaan het hoogteprofiel langs het gevlogen spoor. Zie [Terreinanalyse](#terreinanalyse).
* **Sensoren**: live grafieken van hoogte (ToF, barometer, positie), terrein onder de drone,
  snelheid, houding (pitch/roll), versnelling, batterij en temperatuur. Kies het venster
  (30 s, 1 min, 5 min) en exporteer alles naar CSV voor een verslag.

![Kaart en terrein](docs/gui_terrein.png)

**Rechts**: camerabeeld (voor/onder, met of zonder detecties, opnemen voor de dataset),
de instrumenten (kunstmatige horizon, kompas met de vastgehouden richting, hoogtemeter met de
grond eronder), de belangrijkste sensorwaarden en de gevonden plassen (*Vlieg naar plas*).

![Sensoren](docs/gui_sensoren.png)

De Jetson-verbinding blijft actief terwijl de GUI open is: missies van de Jetson worden
gewoon uitgevoerd en getekend.

## Protocol (JSON over UDP, één bericht per pakket)

**Jetson → drone** (poort 9000)

| type | velden | betekenis |
|---|---|---|
| `mission` | `id`, `waypoints` (`[{"x","y","z"}, …]` of `[[x,y,z], …]`), optioneel `speed` (10-100 cm/s), `land_at_end`, `nav_mode` (`"go"`/`"rc"`), `fine` (`"auto"`/`"all"`/`"last"`/`"off"`) | Opstijgen (als nodig) en de waypoints afvliegen |
| `abort` | – | Missie afbreken na de huidige stap en landen |
| `land` / `takeoff` | – | `land` breekt ook een lopende missie af |
| `set_pose` | `x`, `y`, `yaw` (graden, links = positief) | Startpositie/-richting instellen (alleen op de grond) |
| `get_puddles` | – | Antwoord: `puddle_list` |
| `reset_puddles` | – | Plassenlijst wissen |
| `ping` | `t` | Antwoord: `pong` |

**Drone → Jetson** (poort 9001; naar het IP waar het laatste commando vandaan kwam, of `JETSON_HOST`)

| type | velden |
|---|---|
| `puddle` | `id`, `x`, `y` (cm, mission frame), `area_cm2`, `hits`, `mission`, `confidence` (alleen bij een model) |
| `status` | `state`, `battery`, `pos {x,y,z,yaw}`, `odometry` (true/false), `mission`, `waypoint_index`, `puddles` (elke seconde) |
| `ack` | `ref`, … |
| `waypoint_reached` | `id`, `index`, `pos` |
| `mission_done` / `mission_aborted` | `id`, `puddles` (volledige lijst) |
| `puddle_list` | `puddles` |
| `error` | `ref`, `error` |

UDP kan pakketten verliezen: gebruik `mission_done` / `get_puddles` als de definitieve lijst.

## Hoe het werkt

**Vliegen.** Zie [Nauwkeurig vliegen](#nauwkeurig-vliegen). Vóór het opstijgen worden alle
waypoints gecontroleerd tegen een geofence (`GEOFENCE` in `drone/config.py`).

**Plassen.** Op het zwart-witbeeld van de onderste camera is de vloer het grootste oppervlak,
dus de mediaan-grijswaarde is ongeveer “droge vloer”. Natte plekken zijn duidelijk donkerder
(`PUDDLE_MODE="dark"`) of, als er licht in weerspiegelt, juist feller (`"bright"`/`"both"`).
Na een drempel en wat opkuisen houden we blobs over die groot en compact genoeg zijn.
Blobs die de beeldrand raken worden genegeerd, omdat hun middelpunt dan niet klopt.

**Pixel → coördinaat.** Met de hoogte uit de ToF-sensor en de beeldhoek van de camera wordt een
pixel omgerekend naar cm. Daarna komt de positie van de drone op het moment van het beeld
erbij (min `FRAME_LATENCY_S` vertraging). Een plas wordt pas gemeld als hij `MIN_HITS` keer
gezien is. Detecties binnen `MERGE_RADIUS_CM` worden samengevoegd tot één plas.

## Terreinanalyse

De Tello heeft twee sensoren die samen het terrein kunnen meten:

* de **barometer** meet de luchtdruk, dus de hoogte van de drone t.o.v. de start;
* de **ToF-sensor** onder de drone meet de afstand tot wat eronder ligt.

`grondhoogte = barometer-hoogte − ToF-afstand`. Vliegt de drone over een doos, dan wordt de
ToF-afstand kleiner terwijl de barometer-hoogte gelijk blijft: daar ligt de grond hoger.
De eerste metingen in de lucht bepalen het nulniveau (de vloer bij de start).

De metingen worden per vakje van 20 × 20 cm gemiddeld (`TERRAIN_CELL_CM`). De analyse geeft het
gemeten oppervlak, de laagste/hoogste punten, de ruwheid en een lijst van **obstakels**
(aaneengesloten vakjes hoger dan `TERRAIN_OBSTACLE_CM` = 15 cm) en dalingen. Exporteren kan met
*Terrein → CSV*.

**Nauwkeurigheid**: de barometer ruist (±10–20 cm) en verloopt langzaam met de luchtdruk. Dozen,
treden en hellingen vind je dus wel, kleine hoogteverschillen niet. Verschuift de kaart in de
loop van een vlucht, hang dan boven de vloer en druk op *Herijken*. In `drone/config.py` kun je
met `TERRAIN_ALT_SOURCE = "h"` ook de eigen hoogteschatting van de Tello proberen in plaats van
de barometer.

Let op: boven een obstakel meet de ToF-sensor een kleinere hoogte. De odometrie gebruikt die
hoogte, dus net boven een rand kan ze even “kwijt” zijn (de drone vliegt dan verder in
stap-modus).

In de simulator liggen een doos (30 cm), een plank (12 cm) en een kist (45 cm) op de vloer
(`SIM_TERRAIN` in `drone/sim.py`), zodat je dit zonder drone kunt uitproberen.

## Nauwkeurig vliegen

De Tello heeft geen GPS. Vroeger werd de positie enkel *geschat* door alle commando's op te
tellen. Als de drone tussen twee punten stilhing en afdreef, of door wind draaide, wist de
code dat niet. Nu wordt de echte beweging gemeten (`drone/odometry.py`):

* **Visuele odometrie**: de onderste camera ziet de vloer. De verschuiving van de vloer
  tussen het beeld en een referentiebeeld, maal de grootte van een pixel op de grond (uit de
  hoogte en de beeldhoek), geeft de echte verplaatsing. Ook afdrijven tijdens het stilhangen
  wordt zo gemeten. Werkt het best op een vloer met wat textuur (tegels, hout, tapijt,
  vlekken). Op een egale, glanzende vloer of bij wazige beelden is de odometrie "kwijt" en
  wordt er gerekend met de commando's zoals vroeger. De GUI toont dat bij *Odometrie*.
* **Richting vasthouden**: het kompas (IMU) van de Tello meet de richting. Draait de drone
  weg van de startrichting, dan draait hij terug.
* **Hoogte** komt uit de afstandssensor onder de drone.

**Vliegmodi** (in de GUI bij *Vliegmodus*):

| Modus | Hoe | Voor |
|---|---|---|
| **Stap** (`go`) | Eén `go`-beweging per punt, berekend vanaf de *gemeten* positie. Afdrijven wordt dus bij de volgende beweging rechtgezet. | Betrouwbaar, werkt ook zonder odometrie. De drone stopt kort bij elk punt. |
| **Vloeiend** (`rc`) | De drone wordt 15× per seconde bijgestuurd langs het pad, zonder te stoppen. Wind wordt meteen gecompenseerd. | Vloeiend en nauwkeurig, maar heeft werkende odometrie nodig. Het eerste punt gaat in stap-modus (om de odometrie te controleren). Valt de odometrie weg, dan gaat hij verder in stap-modus. |

**Opstijgen en landen**: een Tello schuift bij het opstijgen en landen vaak 10–30 cm opzij,
ook zonder wind. Met werkende odometrie corrigeert het script dat nu:

* de odometrie blijft meten terwijl de drone stijgt of daalt (vroeger begon ze bij elke
  hoogteverandering opnieuw, waardoor het afschuiven tijdens het opstijgen niet gemeten werd);
* na het opstijgen gaat de drone terug boven de startplek (`TAKEOFF_RECENTER`);
* aan het einde van een missie daalt hij langzaam tot `LAND_HOVER_CM` (30 cm) terwijl hij boven
  het laatste punt blijft, pas daarna volgt de gewone `land` (`PRECISE_LAND`). De knop *Landen*
  en een noodlanding landen meteen, zonder deze stap.

In de simulator (met afschuiven bij opstijgen/landen) ging de afstand tussen start- en
landingsplek bij “1 m vooruit en terug” van 19–59 cm naar 6–25 cm. Werkt de odometrie niet
(GUI: *Odometrie ✖*), dan kan het script het afschuiven niet zien en dus ook niet corrigeren.

**Nauwkeurig positioneren**: de Tello kan geen `go`-beweging kleiner dan 20 cm maken. Met
kleine rc-bijsturingen zet de drone zich daarom tot op `FINE_TOL_CM` (8 cm) op het punt.
*Automatisch* doet dat op elk punt in de stap-modus en op het laatste punt in de vloeiende
modus.

**Resultaat in de simulator** (zelfde parcours, met wind en draaien):

| Modus | Gem. fout per waypoint | Max. fout | Duur |
|---|---|---|---|
| Vroeger (alleen commando's) | 37–48 cm | 48–85 cm | 22 s |
| Stap + odometrie + richting vasthouden | 7–16 cm | 11–24 cm | 29–43 s |
| Vloeiend (rc) + odometrie | 7–11 cm | 10–17 cm | 21–22 s |

In het echt hangt dit af van de vloer (textuur) en van een goede kalibratie van de camera
(zie [Kalibreren](#kalibreren-belangrijk-vóór-de-eerste-echte-vlucht)).

**Veiligheid**: na elke beweging wordt de gemeten verplaatsing vergeleken met het commando.
Is de gemeten beweging even lang maar 90°, 180° of 270° gedraaid, dan is het camerabeeld
gedraaid gemonteerd: dat wordt één keer automatisch gecorrigeerd (`CAM_ROTATE_DEG`). Klopt het
dan nog niet, dan wordt de odometrie uitgeschakeld en vliegt de drone verder zoals vroeger.

**Kleine hoogteverschillen**: de hoogte komt uit de ToF-sensor, die de afstand tot de grond
meet. Een kleine trede of plank in de vloer lijkt dus een hoogteverandering. Verschillen
kleiner dan `Z_DEADBAND_CM` (25 cm) worden niet gecorrigeerd: de drone vliegt gewoon recht
door. Zet het op 0 als je wilt dat hij de hoogte altijd bijstuurt. In de vloeiende modus stopt de drone
ook als hij verder van het punt raakt in plaats van dichter. Draait de richtingcorrectie de
verkeerde kant op, dan schakelt die zichzelf uit (zie `YAW_SIGN`).

## Eigen model (Roboflow, objectdetectie)

Zolang er geen model is, gebruikt het script de drempelmethode (`DETECTOR_BACKEND = "threshold"`).
Een objectdetectiemodel geeft een **bounding box** per plas. Het script neemt het midden van
de box als plaspositie en schat de oppervlakte als een ellips binnen de box.

### 1. Beelden verzamelen

Train op beelden van **dezelfde camera, hoogte en vloer** als tijdens de missie. Dat is
belangrijker dan het aantal beelden.

```bash
python tello_gui.py --record dataset
```

Dit slaat 2 beelden per seconde van de onderste camera op als PNG, al bijgesneden en in
grijswaarden (precies wat het model straks te zien krijgt). Opnemen kun je ook aan en uit
zetten met de knop *Opnemen* in de GUI. Zo kun je een rondje vliegen boven de plassen en
tegelijk opnemen. Vlieg op verschillende hoogtes (bv. 50–120 cm),
met verschillende vormen en groottes van plassen, ander licht, en neem ook beelden **zonder**
plassen op (vlekken, schaduwen, tape, kabels), zodat het model leert wat géén plas is.

### 2. Roboflow

1. Upload de map `dataset/` naar je Roboflow-project (type **Object Detection**).
2. Teken boxen rond de plassen, met één klasse, bv. `puddle`. Beelden zonder plas laat je
   leeg (Roboflow: “mark null”).
3. Preprocessing: Auto-Orient + Resize (bv. 320×320 of 640×640). Gebruik voorzichtige
   augmentations (helderheid, kleine rotaties, flip). Grayscale is al zo.
4. Trainen kan op twee manieren:
   * **In Roboflow (Roboflow Train)** → `DETECTOR_BACKEND = "roboflow"`,
     `ROBOFLOW_MODEL_ID = "<project>/<versie>"` en je API-key in `ROBOFLOW_API_KEY` (of
     als omgevingsvariabele). `pip install inference` downloadt het model één keer.
     **Doe die eerste start terwijl je internet hebt**: op de Wi-Fi van de Tello is er geen internet.
     Daarna draait het model lokaal.
   * **Zelf met Ultralytics**: exporteer de dataset als “YOLOv8”/“YOLO11” en train, bv. in Colab:
     `yolo detect train data=data.yaml model=yolo11n.pt imgsz=320 epochs=100`.
     Zet `best.pt` in `models/puddles.pt` → `DETECTOR_BACKEND = "yolo"`, `MODEL_IMGSZ = 320`.
     Dit werkt volledig offline, en later kun je hetzelfde model ook op de Jetson draaien.

### 3. Testen en afstellen

```bash
python -m drone.puddle_detector dataset/ --backend yolo   # map met beelden doorlopen
python tello_gui.py --sim                                 # (sim tekent eenvoudige plassen)
```

Instellingen in `drone/config.py`: `MODEL_CONFIDENCE` (hoger = minder valse meldingen),
`MODEL_CLASSES` (bv. `["puddle"]`), `MIN_HITS` (hoe vaak een plas gezien moet zijn) en
`REJECT_BORDER_BLOBS` (boxen tegen de beeldrand overslaan, omdat de plas dan half in beeld is).
Kies een klein model (`n`), want het draait op de laptop-CPU. De detectie probeert tot
`DETECT_HZ` = 10 keer per seconde te draaien. Een plas moet `MIN_HITS` keer gezien
worden terwijl hij volledig in beeld is, dus vlieg trager als het model traag is.

## Kalibreren (belangrijk vóór de eerste echte vlucht)

1. **Beeld controleren.** Start `tello_combined.py`, zet downvision aan (V) en maak een foto (F).
   Kijk of het onderste camerabeeld in zwarte randen zit. `AUTO_CROP` vangt dat normaal op;
   anders zet je `DOWNCAM_CROP = (x, y, w, h)`.
2. **Beeldhoek (`CAM_HFOV_DEG`).** Houd de drone op een bekende hoogte `h` (cm) boven een
   meetlat en lees af hoeveel cm `W` er horizontaal in beeld past:
   `CAM_HFOV_DEG = 2 * atan((W / 2) / h)` (in graden).
3. **Oriëntatie.** Leg een donker voorwerp **vóór** de drone en schuif de drone er langzaam
   naartoe. Komt het voorwerp van **boven** het beeld binnen? Dan `CAM_FORWARD_SIGN = 1`, anders `-1`.
   Doe hetzelfde naar links voor `CAM_LEFT_SIGN`.
   Komt het voorwerp van **opzij** binnen terwijl je vooruit schuift, dan is het beeld 90°
   gedraaid: zet `CAM_ROTATE_DEG` (0, 90, 180 of 270). Dat wordt ook automatisch gemeten bij de
   eerste beweging van een vlucht; de log toont dan *Camerabeeld is … gedraaid* met de waarde
   die je in `drone/config.py` moet zetten.
4. **Detectie afstellen.** Maak foto's/video van echte plassen op jullie vloer en run
   `python -m drone.puddle_detector foto.jpg`: je ziet de gevonden plassen en het masker. Pas
   `DARK_OFFSET`, `MIN_AREA_PX`, `MIN_SOLIDITY` en `PUDDLE_MODE` aan tot het klopt.
5. **Richting (`YAW_SIGN`).** Zet de drone aan, open de GUI en draai de drone met de hand naar
   **links**: de *Richting* in de status moet **stijgen**. Daalt hij, zet dan `YAW_SIGN = 1`.
6. **Odometrie.** Vlieg met de stap-modus een punt 1 m vooruit. Klopt de camera-oriëntatie
   (stap 3) niet, dan zie je in de log "odometrie UITGESCHAKELD". Klopt de afstand niet
   (bv. de positie zegt 80 cm terwijl hij 1 m vloog), dan staat `CAM_HFOV_DEG` (stap 2)
   verkeerd: de odometrie gebruikt dezelfde beeldhoek om pixels naar cm om te rekenen.
7. **Grote plassen.** De camera ziet op 80 cm hoogte maar ongeveer 90 × 70 cm (met 60° beeldhoek).
   Plassen die bijna zo groot zijn als het beeld raken bijna altijd de rand en worden dan niet
   gemeld. Vlieg hoger of zet `REJECT_BORDER_BLOBS = False` (dan minder nauwkeurige middelpunten).

## Veiligheid

* Eerst testen met `--sim`, daarna met een kleine missie (`json_flights/mission_example.json`) in een lege ruimte.
* De missie wordt geweigerd als de batterij onder `MIN_BATTERY` zit of een waypoint buiten de geofence ligt.
* Bij een fout tijdens de missie landt de drone automatisch.
* De drone heeft genoeg textuur op de vloer nodig om stabiel te hangen. Een spiegelende natte
  vloer kan de optische flow storen: test dat voorzichtig.
* Alleen de missiethread stuurt vliegcommando's naar de Tello (djitellopy is niet thread-safe).
