# Tello EDU – autonoom pad volgen + plassen detecteren

```
tello_gui.py          ← start hier: grafische besturing (coördinaten, 3D-pad, camera, plassen)
tello_combined.py     ← live vliegen met het toetsenbord (ZQSD), handig om te testen/kalibreren
drone/
    app.py            kern: missies afvliegen, plassen detecteren, Jetson-koppeling
    config.py         alle instellingen (netwerk, snelheid, geofence, camera, detectie)
    navigator.py      positieschatting (dead reckoning) + waypoints afvliegen met `go x y z`
    puddle_detector.py  plasdetectie (drempelmethode of getraind model) + samenvoegen
    jetson_link.py    UDP/JSON-communicatie met de Jetson
    sim.py            simulator: alles testen zonder drone (`--sim`)
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

* **Waypoints**: vul x, y en z in (cm) en druk op Enter of *Toevoegen*. Klik een punt in de
  lijst aan om het aan te passen (*Bijwerken*), te verplaatsen (▲▼) of te verwijderen.
  *Huidige positie* neemt de positie van de drone over. Elk punt wordt meteen gecontroleerd
  tegen de geofence.
* **Opslaan/Laden**: vluchten worden als JSON bewaard in `json_flights/`
  (zie `json_flights/mission_example.json` voor het formaat).
* **3D-rooster**: oranje = ingevoerd pad (genummerde punten met coördinaten), blauw = actieve
  missie (ook missies van de Jetson), groen = bereikte punten, rood = gevlogen spoor, rode X =
  drone (met stippellijn naar de vloer), cyaan = gevonden plassen. Sleep met de muis om te
  draaien of kies *3D*, *Boven* of *Zijkant*.
* **Vliegen**: *Start missie*, *Ga naar geselecteerd punt* (blijft daarna hangen),
  *Opstijgen*, *Landen* (toets L) en **NOODSTOP** (toets X, motoren uit, de drone valt!).
* Onderaan: het beeld van de onderste camera met detecties, de gevonden plassen, de log en
  een knop om beelden op te nemen voor de dataset.

De Jetson-verbinding blijft actief terwijl de GUI open is: missies van de Jetson worden
gewoon uitgevoerd en in het rooster getekend.

## Protocol (JSON over UDP, één bericht per pakket)

**Jetson → drone** (poort 9000)

| type | velden | betekenis |
|---|---|---|
| `mission` | `id`, `waypoints` (`[{"x","y","z"}, …]` of `[[x,y,z], …]`), optioneel `speed` (10-100 cm/s), `land_at_end` | Opstijgen (als nodig) en de waypoints afvliegen |
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
| `status` | `state`, `battery`, `pos {x,y,z,yaw}`, `mission`, `waypoint_index`, `puddles` (elke seconde) |
| `ack` | `ref`, … |
| `waypoint_reached` | `id`, `index`, `pos` |
| `mission_done` / `mission_aborted` | `id`, `puddles` (volledige lijst) |
| `puddle_list` | `puddles` |
| `error` | `ref`, `error` |

UDP kan pakketten verliezen: gebruik `mission_done` / `get_puddles` als de definitieve lijst.

## Hoe het werkt

**Vliegen.** De Tello heeft geen GPS. `go x y z speed` gebruikt de optische-flowsensor
onderaan en is redelijk nauwkeurig. `navigator.py` telt alle bewegingen op tot een
positieschatting en splitst lange stukken op in stappen van max. `MAX_STEP_CM` (standaard 1 m).
Daardoor zijn de plasposities nauwkeuriger en reageert een `abort` sneller. De Tello kan
geen beweging maken waarbij x, y én z allemaal kleiner dan 20 cm zijn. Zo'n restfout gaat
niet verloren: het volgende waypoint corrigeert ervoor. Vóór het opstijgen worden alle
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
4. **Detectie afstellen.** Maak foto's/video van echte plassen op jullie vloer en run
   `python -m drone.puddle_detector foto.jpg`: je ziet de gevonden plassen en het masker. Pas
   `DARK_OFFSET`, `MIN_AREA_PX`, `MIN_SOLIDITY` en `PUDDLE_MODE` aan tot het klopt.
5. **Grote plassen.** De camera ziet op 80 cm hoogte maar ongeveer 90 × 70 cm (met 60° beeldhoek).
   Plassen die bijna zo groot zijn als het beeld raken bijna altijd de rand en worden dan niet
   gemeld. Vlieg hoger of zet `REJECT_BORDER_BLOBS = False` (dan minder nauwkeurige middelpunten).

## Veiligheid

* Eerst testen met `--sim`, daarna met een kleine missie (`json_flights/mission_example.json`) in een lege ruimte.
* De missie wordt geweigerd als de batterij onder `MIN_BATTERY` zit of een waypoint buiten de geofence ligt.
* Bij een fout tijdens de missie landt de drone automatisch.
* De drone heeft genoeg textuur op de vloer nodig om stabiel te hangen. Een spiegelende natte
  vloer kan de optische flow storen: test dat voorzichtig.
* Alleen de missiethread stuurt vliegcommando's naar de Tello (djitellopy is niet thread-safe).
