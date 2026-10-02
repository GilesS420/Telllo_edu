# Tello EDU – autonoom pad volgen + plassen detecteren

| Bestand | Wat |
|---|---|
| `tello_combined.py` + `KeyPressModule.py` | Handmatige besturing met het toetsenbord (ongewijzigd) |
| `tello_gui.py` | **Grafische besturing**: coördinaten invullen, 3D-rooster met pad, drone en plassen |
| `tello_autonomous.py` | **Hoofdscript**: vliegt een pad (van de Jetson of handmatig ingevuld) en meldt plassen |
| `manual_input.py` | Console om zonder Jetson coördinaten in te typen |
| `navigator.py` | Positieschatting (dead reckoning) + waypoints afvliegen met `go x y z speed` |
| `puddle_detector.py` | Plasdetectie op de onderste camera (drempelmethode of getraind model) + samenvoegen van detecties |
| `jetson_link.py` | UDP/JSON-communicatie met de Jetson |
| `jetson_client_example.py` | Voorbeeld voor de **Jetson-kant** (pad sturen, plassen ontvangen, coördinaten terugrekenen) |
| `sim.py` | Simulator: alles testen zonder drone (`--sim`) |
| `config.py` | Alle instellingen (netwerk, snelheid, geofence, camera-kalibratie, detectie) |
| `mission_example.json` | Voorbeeldmissie (vierkant van 1 × 1 m) |

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

De Jetson rekent echte-wereldcoördinaten om naar dit stelsel (zie `world_to_drone()` in
`jetson_client_example.py`) en rekent de plascoördinaten die terugkomen weer om (`drone_to_world()`).
Staat de drone gedraaid ten opzichte van het wereldstelsel van de Jetson, dan kan dat op twee manieren:
in `world_to_drone()` (START_HEADING_DEG) **of** door vóór het opstijgen
`{"type":"set_pose","x":..,"y":..,"yaw":..}` te sturen.

## Installatie (in je venv)

```bash
pip install -r requirements.txt
# pas nodig als jullie model klaar is (kies één):
pip install ultralytics   # DETECTOR_BACKEND = "yolo"
pip install inference     # DETECTOR_BACKEND = "roboflow"
```

## Gebruik

```bash
python tello_autonomous.py --sim                           # testen zonder drone
python tello_autonomous.py                                 # echte drone: Jetson én handmatige console
python tello_autonomous.py --waypoints "100,0,80; 100,100,80; 0,0,80"   # pad meegeven, vliegt meteen
python tello_autonomous.py --mission mission_example.json  # missie uit een bestand, vliegt meteen
python tello_autonomous.py --record dataset                # camerabeelden opslaan voor Roboflow
```

`--speed 20` verandert de snelheid van `--waypoints`, en met `--hover` blijft de drone na het
laatste punt hangen in plaats van te landen.

### Grafische besturing (aanrader)

```bash
python tello_gui.py --sim     # eerst proberen zonder drone
python tello_gui.py           # echte drone (laptop op de Wi-Fi van de Tello)
```

![Tello GUI](docs/gui.png)

* **Waypoints**: vul x, y en z in (cm) en druk op Enter of *Toevoegen*. Klik een punt in de
  lijst aan om het aan te passen (*Bijwerken*), te verplaatsen (▲▼) of te verwijderen.
  *Huidige positie* neemt de positie van de drone over. Een pad kan je *Opslaan*/*Laden* als
  JSON, in hetzelfde formaat als `mission_example.json`.
* **3D-rooster**: oranje = ingevoerd pad (genummerde punten met coördinaten), blauw = actieve
  missie (ook missies van de Jetson), groen = bereikte punten, rood = gevlogen spoor, rode X =
  drone (met stippellijn naar de vloer), cyaan = gevonden plassen. Sleep met de muis om te
  draaien of kies *3D*, *Boven* of *Zijkant*.
* **Vliegen**: *Start missie*, *Ga naar geselecteerd punt* (blijft daarna hangen),
  *Opstijgen*, *Landen* (toets L) en **NOODSTOP** (toets X, motoren uit, de drone valt!).
* Onderaan: het beeld van de onderste camera met detecties, de gevonden plassen en de log.

De Jetson-verbinding blijft actief terwijl de GUI open is.

### Handmatig coördinaten invullen in de terminal (zonder Jetson)

Na het opstarten kun je in de terminal commando's typen (`help` toont ze allemaal):

```
tello> 100 0 80          # waypoint toevoegen: x=100 cm vooruit, y=0, hoogte 80 cm
tello> 100 100 80        # 1 m vooruit en 1 m naar links
tello> 0 0 80            # terug boven het startpunt
tello> list              # controleren   (undo = laatste weg, clear = alles weg)
tello> start             # opstijgen, afvliegen en landen   (start hover = blijven hangen)
tello> goto 50 0 100     # vanuit de huidige positie direct naar één punt, blijven hangen
tello> pos               # waar denkt de drone dat hij is?
tello> puddles           # gevonden plassen
tello> land
```

Elk punt wordt meteen gecontroleerd tegen de geofence. De handmatige console en de Jetson
gebruiken exact dezelfde vliegcode, dus wat nu handmatig werkt, werkt straks ook via de Jetson.

Op de Jetson (of op dezelfde laptop om te testen):

```bash
python jetson_client_example.py --drone <ip-van-de-laptop>
```

Toetsen in het videovenster: **L/P** = landen (missie afbreken), **R** = opnemen aan/uit,
**X** = noodstop (motoren uit, de drone valt!), **ESC** = landen en afsluiten.

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
waypoints gecontroleerd tegen een geofence (`GEOFENCE` in `config.py`).

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
python tello_autonomous.py --record dataset
```

Dit slaat 2 beelden per seconde van de onderste camera op als PNG, al bijgesneden en in
grijswaarden (precies wat het model straks te zien krijgt). Opnemen kun je ook aan en uit
zetten met **R** of `record` in de console. Zo kun je met de console een rondje vliegen
boven de plassen en tegelijk opnemen. Vlieg op verschillende hoogtes (bv. 50–120 cm),
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
python puddle_detector.py dataset/ --backend yolo      # map met beelden doorlopen
python tello_autonomous.py --sim                       # (sim tekent eenvoudige plassen)
```

Instellingen in `config.py`: `MODEL_CONFIDENCE` (hoger = minder valse meldingen),
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
   `python puddle_detector.py foto.jpg`: je ziet de gevonden plassen en het masker. Pas
   `DARK_OFFSET`, `MIN_AREA_PX`, `MIN_SOLIDITY` en `PUDDLE_MODE` aan tot het klopt.
5. **Grote plassen.** De camera ziet op 80 cm hoogte maar ongeveer 90 × 70 cm (met 60° beeldhoek).
   Plassen die bijna zo groot zijn als het beeld raken bijna altijd de rand en worden dan niet
   gemeld. Vlieg hoger of zet `REJECT_BORDER_BLOBS = False` (dan minder nauwkeurige middelpunten).

## Veiligheid

* Eerst testen met `--sim`, daarna met een kleine missie (`mission_example.json`) in een lege ruimte.
* De missie wordt geweigerd als de batterij onder `MIN_BATTERY` zit of een waypoint buiten de geofence ligt.
* Bij een fout tijdens de missie landt de drone automatisch.
* De drone heeft genoeg textuur op de vloer nodig om stabiel te hangen. Een spiegelende natte
  vloer kan de optische flow storen: test dat voorzichtig.
* Alleen de missiethread stuurt vliegcommando's naar de Tello (djitellopy is niet thread-safe).
