# Tello EDU – autonoom pad volgen + plassen detecteren

| Bestand | Wat |
|---|---|
| `tello_combined.py` + `KeyPressModule.py` | Handmatige besturing met het toetsenbord (ongewijzigd) |
| `tello_autonomous.py` | **Hoofdscript**: ontvangt een pad van de Jetson, vliegt het af en meldt plassen |
| `navigator.py` | Positieschatting (dead reckoning) + waypoints afvliegen met `go x y z speed` |
| `puddle_detector.py` | Plasdetectie op de onderste camera (downvision) + samenvoegen van detecties |
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
```

## Gebruik

```bash
python tello_autonomous.py --sim                         # testen zonder drone
python tello_autonomous.py                               # echte drone, wacht op de Jetson
python tello_autonomous.py --mission mission_example.json  # echte drone, missie uit een bestand
```

Op de Jetson (of op dezelfde laptop om te testen):

```bash
python jetson_client_example.py --drone <ip-van-de-laptop>
```

Toetsen in het videovenster: **L/P** = landen (missie afbreken), **X** = noodstop (motoren uit,
de drone valt!), **ESC** = landen en afsluiten.

## Protocol (JSON over UDP, één bericht per pakket)

**Jetson → drone** (poort 9000)

| type | velden | betekenis |
|---|---|---|
| `mission` | `id`, `waypoints` (`[{"x","y","z"}, …]` of `[[x,y,z], …]`), optioneel `speed` (10-100 cm/s), `land_at_end` | Opstijgen (als nodig) en de waypoints afvliegen |
| `abort` | – | Missie afbreken na de huidige stap en landen |
| `land` / `takeoff` | – | |
| `set_pose` | `x`, `y`, `yaw` (graden, links = positief) | Startpositie/-richting instellen (alleen op de grond) |
| `get_puddles` | – | Antwoord: `puddle_list` |
| `reset_puddles` | – | Plassenlijst wissen |
| `ping` | `t` | Antwoord: `pong` |

**Drone → Jetson** (poort 9001; naar het IP waar het laatste commando vandaan kwam, of `JETSON_HOST`)

| type | velden |
|---|---|
| `puddle` | `id`, `x`, `y` (cm, mission frame), `area_cm2`, `hits`, `mission` |
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
