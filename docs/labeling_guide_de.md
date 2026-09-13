# Labeling-Guide: Diagrammtypen in Geschäftsberichten

**Zweck:** Einheitliche Vergabe von Klassenlabels für Bildausschnitte (Figures), die
automatisch aus PDF-Geschäftsberichten extrahiert wurden. Die Labels dienen als
Trainings- und Validierungsdaten für einen Bildklassifikator.

**Grundregel:** Ein Bild bekommt **genau ein** Label. Im Zweifel entscheidet der
Entscheidungsbaum in Abschnitt 2, nicht das Bauchgefühl.

> Massgebliche Fassung für das Labeling-Team. Englische Übersetzung:
> [labeling_guide_en.md](labeling_guide_en.md).

---

## 1. Klassenübersicht

| Label | Deutsch | Kurzdefinition |
|---|---|---|
| `bar` | Balken-/Säulendiagramm | Einfache Balken einer Reihe auf gemeinsamer Grundlinie, senkrecht oder waagerecht |
| `bar_grouped` | Gruppierte Säulen/Balken | Mehrere Balken je Kategorie nebeneinander, einer pro Reihe |
| `bar_stacked` | Gestapelte Säulen/Balken | Segmente innerhalb eines Balkens aufeinandergesetzt |
| `waterfall` | Wasserfall / Brücke | Schwebende Balken zwischen Start- und Endwert |
| `line` | Liniendiagramm | Werte als verbundener Linienzug |
| `combo_bar_line` | Kombidiagramm | Balken **und** Linie in einem Plot |
| `scatter` | Streudiagramm | Punkte in einem X/Y-Koordinatensystem, ohne verbindende Linie |
| `pie_donut` | Kreis-/Ringdiagramm | Kreisförmige Anteilsdarstellung |
| `flow` | Flussdiagramm | Knoten durch **gezeichnete** Verbindungen zu Ablauf oder Hierarchie verbunden |
| `map` | Karte | Geografische Darstellung |
| `table` | Tabelle | Zeilen-/Spaltenraster ohne grafische Kodierung |
| `photo` | Foto | Fotografische Aufnahme |
| `other` | Sonstiges | Logos, Siegel, Icons/Piktogramme und alles, was in keine Klasse oben passt |

---

## 2. Entscheidungsbaum

Von oben nach unten durchgehen, beim ersten Treffer stoppen.

```
1. Ist es eine Fotografie (Personen, Gebäude, Produkte)?      → photo
2. Ist es ein Logo, Siegel, Award oder Piktogramm?            → other
3. Zeigt das Bild Diagramme MEHRERER VERSCHIEDENER Typen?     → Regel R1 (Abschnitt 4)
4. Ist eine geografische Karte das dominante Element?         → map
5. Knoten durch GEZEICHNETE Verbindungen zu Pfad/Hierarchie?   → flow
   (Radial/Nabe-Speiche, Ringe, Stapel ohne Verbindungen       → other)
6. Sind Balken UND eine Linie als Datenreihen vorhanden?      → combo_bar_line
7. Kreisförmige Anteilsdarstellung (Voll- oder Ringform)?     → pie_donut
8. Schweben Balken zwischen Start- und Endwert (Brücke)?      → waterfall
9. Sind Balken in Segmente unterteilt (gestapelt)?            → bar_stacked
10. Mehrere Balken je Kategorie nebeneinander (gruppiert)?    → bar_grouped
11. Einfache Balken einer Reihe (senkrecht oder waagerecht)?  → bar
12. Punkte in einem X/Y-System ohne verbindende Linie?        → scatter
13. Ist es ein Linienzug (auch Fläche darunter)?             → line
14. Reines Zeilen-/Spaltenraster ohne Grafik?                → table
15. Sonst                                                    → other
```

Die Reihenfolge ist bewusst gewählt: `combo_bar_line` (Schritt 6) steht vor
waterfall/stacked/grouped/bar/line, weil ein Kombidiagramm sonst je nach
Blickwinkel als Balken- oder Liniendiagramm gelabelt würde. `bar_grouped`
(Schritt 10) wird vor dem einfachen `bar` geprüft und `scatter` (Schritt 12) vor
`line` — in beiden Paaren geht das erste sonst leicht als das zweite durch.

---

## 3. Klassen im Detail

### `bar` — Balken-/Säulendiagramm

Einfache Balken **einer Datenreihe**, alle auf derselben Grundlinie beginnend.
Die Orientierung — senkrecht (Säulen) oder waagerecht (Balken) — wird **nicht**
unterschieden: Sie ist eine reine Layout-Entscheidung ohne nachgelagerte
Bedeutung, beide Formen werden gleich geparst.

**Typisch:** Umsatz je Geschäftsjahr, EBIT-Entwicklung, Mitarbeiterzahl,
F&E-Aufwand, Investitionen; Umsatz nach Region, Top-10-Rankings, Altersstruktur
der Belegschaft, Umfrageergebnisse.

**Gehört ebenfalls hierher:**
- Senkrechte (Säulen) **und** waagerechte (Balken) Ausrichtung.
- Balken mit Datenbeschriftung statt Achse.
- Einzelner hervorgehobener Balken (Farbakzent).

**Nicht hierher:**
- Mehrere Reihen als benachbarte Balken je Kategorie → `bar_grouped`
- Balken in Segmente unterteilt → `bar_stacked`
- Balken beginnen nicht auf der Grundlinie → `waterfall`
- Zusätzliche Linie als Datenreihe → `combo_bar_line`

### `bar_grouped` — Gruppiertes Balken-/Säulendiagramm

Mehrere Balken je Kategorie, nebeneinander gestellt — ein Balken pro Datenreihe,
nicht gestapelt. Wie bei `bar` wird die Orientierung (senkrecht oder waagerecht)
nicht unterschieden.

**Typisch:** Vorjahr vs. Berichtsjahr nebeneinander, Ist vs. Plan,
Regionenvergleich über zwei bis drei Perioden, Mehrreihen-KPI-Vergleiche.

**Gehört ebenfalls hierher:**
- Zwei oder mehr Reihen als benachbarte Balken innerhalb jeder Kategoriegruppe.
- Senkrecht (gruppierte Säulen) wie waagerecht (gruppierte Balken).

**Nicht hierher:**
- Einzelne Reihe einfacher Balken → `bar`
- Reihen innerhalb eines Balkens gestapelt → `bar_stacked`
- Linie als zusätzliche Reihe → `combo_bar_line`

**Warum getrennt von `bar`?** Gruppierte Balken sind schwerer zu parsen: Je
Kategorie stehen mehrere Balken, die über Legende und Farben der richtigen Reihe
zugeordnet werden müssen, und Wertbeschriftungen drängen sich. Die Auslagerung
gibt diesem schwereren Fall eine eigene Route — dieselbe Begründung wie bei
`bar_stacked`.

### `bar_stacked` — Gestapelte Darstellung

Balken sind in farbige Segmente unterteilt, die aufeinander aufbauen.

**Typisch:** Segmentanteile über mehrere Jahre, Umsatzzusammensetzung,
Belegschaft nach Region, 100 %-Aufteilungen.

**Gehört ebenfalls hierher:**
- 100 %-gestapelte Varianten (Skala endet bei 100 %).
- Sowohl senkrechte als auch waagerechte Stapel. Orientierung wird hier
  **nicht** unterschieden.

**Wichtigstes Erkennungsmerkmal:** Alle Segmente eines Balkens teilen sich eine
gemeinsame Grundlinie und liegen lückenlos aufeinander.

**Warum getrennt von `bar`?** Anders als senkrecht/waagerecht wird der Stapel
bewusst als eigene Klasse geführt: Er ist nachgelagert deutlich schwerer zu
parsen (überlappende Segmente, Werte nur als Summe ablesbar) und bekommt eine
eigene Verarbeitungsroute.

### `waterfall` — Wasserfall- / Brückendiagramm

Balken „schweben" zwischen Start- und Endwert und zeigen eine Überleitung.

**Typisch:** EBIT-Brücke, Umsatzbrücke (Volumen / Preis / Währung / M&A),
Überleitung Nettofinanzschulden, Cashflow-Überleitung, Ergebnisveränderung
Vorjahr → Berichtsjahr.

**Erkennungsmerkmale — mindestens zwei sollten zutreffen:**
1. Erster und letzter Balken stehen auf der Grundlinie, die dazwischen nicht.
2. Verbindungslinien zwischen den Balkenenden.
3. Zweifarbige Kodierung für Zu- und Abnahmen (oft grün/rot oder hell/dunkel).
4. Beschriftungen mit Vorzeichen (`+12`, `−8`) an den Zwischenbalken.
5. Achsenbeschriftungen als Veränderungsgründe statt als Kategorien oder Jahre.

**Häufigster Fehler:** Verwechslung mit `bar_stacked`. Siehe Regel R3.

### `line` — Liniendiagramm

Datenpunkte durch einen Linienzug verbunden.

**Typisch:** Aktienkursentwicklung gegen DAX/MDAX, Mehrjahrestrends,
Zinsentwicklung, Absatzverlauf.

**Gehört ebenfalls hierher:**
- Mehrere Linien in einem Plot.
- Flächendiagramme (Linie mit eingefärbter Fläche darunter), auch gestapelt.
  Vorerst zusammengefasst — bei relevantem Volumen später als `area` abtrennen.
- Linien mit Punktmarkern.

**Hinweis — Datenreihen zählen, nicht Striche.** Gitterlinien, Achsen, Ziel- /
Referenzlinien, Ablese-Hilfslinien und Callout-Linien sind Diagramm-Beiwerk und
ändern den Typ nicht. Ein Diagramm ist `line`, wenn es ≥1 Datenreihe als
verbundene Linie in einem x/y-System zeigt — egal, wie viele Hilfsstriche darum
liegen (z. B. eine Vergütungs-Zielerreichungskurve mit Ist-Wert-Marker → `line`).

**Nicht hierher:** Linie zusammen mit Balken → `combo_bar_line`.

### `combo_bar_line` — Kombidiagramm

Balken und Linie als **inhaltliche Datenreihen** in einem Plot, meist mit
Sekundärachse.

**Typisch:** Umsatz (Säulen) plus EBIT-Marge in % (Linie); Absatz plus
Durchschnittspreis; Investitionen plus Investitionsquote.

**Nicht hierher:**
- Zielwert- oder Durchschnittslinie über einem Balkendiagramm. Eine
  Referenzlinie ist keine Datenreihe → bleibt `bar_vertical` /
  `bar_horizontal`.
- Verbindungslinien in einer Brücke → `waterfall`.

**Faustregel:** **Verändert sich die Linie über die Kategorien hinweg?** Dann
ist sie eine Datenreihe → `combo_bar_line`. Eine Linie auf konstantem Wert ist
eine Referenzlinie und ändert die Klasse nicht — **auch dann nicht, wenn sie
einen eigenen Legendeneintrag hat**.

Die Legende ist nicht das Kriterium. Ziel- und Durchschnittslinien stehen
regelmäßig in der Legende („Ziel 2030", „Ø"), weshalb die frühere Formulierung
(„eigene Achse oder eigener Legendeneintrag") im Widerspruch zu der Regel
stand, dass eine Zielwertlinie keine Datenreihe ist. Wenn beide Signale
auseinanderfallen, entscheidet *verändert sie sich*.

### `scatter` — Streudiagramm

Datenpunkte in einem X/Y-Koordinatensystem, **ohne** verbindende Linie zwischen
den Punkten.

**Typisch:** Risiko-Rendite-Streuung, Korrelationsdarstellungen,
Positionierungs- und Portfoliodiagramme mit zwei Wertachsen, Bubble-Charts
(Punkte mit zusätzlicher Größenkodierung).

**Nicht hierher:**
- Punkte durch eine Linie verbunden → `line`.
- Punkte auf einer Karte statt in einem Achsensystem → `map`.

**Hinweis:** Scatter wird vorerst nicht inhaltlich geparst. Die Klasse existiert,
um Streudiagramme zuverlässig von `line` zu trennen — beide sehen sich ähnlich
(Achsen, Punktmarker), und eine Verwechslung würde ein Liniendiagramm
fälschlich zum Parsen schicken.

### `pie_donut` — Kreis- und Ringdiagramm

Anteile als Kreissegmente.

**Typisch:** Aktionärsstruktur, Umsatz nach Region oder Sparte,
Belegschaftsstruktur, Streubesitz.

**Gehört ebenfalls hierher:**
- Ringdiagramme (Donut), auch mit Zahl in der Mitte.
- Halbkreis- und Tachodarstellungen.
- Kreise mit herausgezogenem Segment.

Voll- und Ringform werden **nicht** unterschieden: identische Semantik, und das
Corporate Design wechselt willkürlich zwischen beiden.

### `map` — Karte

Geografische Darstellung als dominantes Element.

**Typisch:** Standortübersicht, Produktionsnetz, Vertriebsregionen, Umsatz nach
Land als Choropleth.

**Auch hierher:** Karten mit eingeblendeten Zahlen, Pins oder kleinen Balken —
solange die Karte den Bildeindruck bestimmt.

### `flow` — Flussdiagramm

Einzelne Knoten, durch **gezeichnete** Verbindungen zu einem gerichteten Ablauf
oder einer Hierarchie verbunden.

**Alle Bedingungen müssen erfüllt sein:**

1. **Explizite, gezeichnete Struktur.** Entweder sind Pfeile/Linien *zwischen*
   den Elementen sichtbar, oder die Elementformen selbst kodieren den Ablauf —
   ineinandergreifende Chevrons, ein segmentierter Pfeil, sich verjüngende
   Trichterstufen. Schlichte Rechtecke, die nur neben- oder übereinander
   liegen, genügen **nicht**.
2. **Richtung oder Hierarchie.** Die Struktur beschreibt einen Ablauf
   (Anfang → Ende) oder einen Baum Eltern → Kind. Man kann einem Pfad folgen.
3. **Das Layout ist linear oder verzweigt — niemals kreisförmig.** Siehe unten.

**Typisch:** Prozess- und Ablaufdiagramme, Entscheidungsbäume, Organigramme und
Governance-Strukturen *mit gezeichneten Berichtslinien*, Wertschöpfungsketten,
Swimlane-Diagramme, Trichter.

**Gehört ebenfalls hierher:**
- Horizontale wie vertikale Flussrichtung.
- Knoten mit Symbolen oder Icons darin — solange Knoten und Verbindungen die
  Struktur bilden.

**Nicht hierher:**
- **Alles Kreisförmige → `other`. `flow` ist niemals rund.** Diese Regel gilt
  absolut und wird vor allen anderen Tests angewandt, damit „kreisförmig" ein
  verlässliches Signal für *nicht* `flow` ist. Sie umfasst:
  - **Radial- und Nabe-Speiche-Diagramme** — ein zentrales Element mit ringsum
    angeordneten Elementen, auch wenn Speichen gezeichnet sind. Es gibt keinen
    Ablauf — die Beziehung lautet „gehört zur Mitte", nicht „geht voran".
  - **Ring-, Rad- und Kreissegment-Grafiken**, bei denen der Kreis selbst die
    Grafik ist. Bei Anteilsdarstellung stattdessen `pie_donut` prüfen.
  - **Prozesskreisläufe** — Kästen im Kreis mit Pfeilen (A → B → C → A), auch
    wenn sie einen Ablauf zeigen. Sie sind kein Parsing-Ziel, und ihr
    Ausschluss hält die Regel frei von Ausnahmen.
- **Gestapelte oder geschichtete Kästen ohne Verbindungen → `other`.**
  Governance-, Compliance- und Organigramme als gestapelte Farbbänder oder
  aneinandergrenzende Kästen ohne gezeichnete Linien. Ebenso geschichtete
  Tempel-/Framework-Diagramme und Pyramiden. Wenn Hierarchie nur dadurch
  angedeutet wird, dass ein Kasten über einem anderen liegt oder ihn
  überspannt, ist es **kein** `flow`.
- Eine Reihe gleichrangiger Kästen ohne Hierarchie oder Reihenfolge — eine
  gestaltete Liste oder ein Icon-Raster, deren einzige Beziehung „gleiche
  Kategorie" ist → `other` (oder `table`, wenn es ein echtes Raster ist).
- Sankey-Diagramme (Flüsse mit proportionaler Breite) → vorerst `other`.
- Zeitstrahlen/Roadmaps ohne verbindende Ablauflogik → `other`. Test: Zeitachse
  entfernen — bleibt nur „Ereignisse in Datumsreihenfolge", ist es ein
  Zeitstrahl (`other`); überlebt eine Beziehungsstruktur (eine Phase bedingt/
  ermöglicht die nächste, Verzweigungen), ist es `flow`.

**Schnelltest:** Ist es rund? → `other`. Sonst die Verbindungen mit der Hand
verdecken: Ist dann nicht mehr erkennbar, was wohin führt, ist es `flow`; trägt
das Layout allein die ganze Aussage, ist es `other`.

**Hinweis:** Das Basismodell (DocumentFigureClassifier) kennt Flussdiagramme
bereits, weshalb weniger echte Trainingsbeispiele nötig sind als für eine ganz
neue Klasse. Die Klasse ist als späteres Parsing-Ziel vorgesehen.

### `table` — Tabelle

Reines Zeilen-/Spaltenraster ohne grafische Kodierung von Werten.

**Typisch:** Kennzahlenübersicht, Mehrjahresvergleich oder ein qualitatives
Raster (ESG-/Ziel-Tabelle, Governance-Grundsätze), das als Bild statt als
Text im PDF vorliegt. Zellen dürfen **Zahlen oder Text** enthalten — nicht der
Inhalt entscheidet über die Klasse, sondern die Struktur.

**Hinweis — verbundene Zellen und unsaubere Struktur zählen mit.** Der Test
lautet „Sieht ein Leser ein Zeilen-/Spaltenraster?", nicht „Ließe sich das
sauber in ein Raster serialisieren?". Verbundene / zeilenübergreifende Zellen,
leere Zellen, Zwischensummen- oder Summenzeilen und über mehrere Zeilen
gespannte Fließtext-Zellen sind normale Tabellenbestandteile → weiterhin
`table`. Die saubere Parsebarkeit ist Aufgabe des nachgelagerten Extraktors und
wird bewusst nicht im Label kodiert.

**Nicht hierher:** Tabellen mit eingebetteten Balken, Ampeln oder Sparklines —
diese als `other` labeln und in der Notizspalte vermerken. Ebenso farbcodierte
Raster wie Risikomatrizen / Heatmaps (die Zellen tragen den Wert über die Farbe,
nicht über Text) → `other`.

### `photo` — Fotografie

**Typisch:** Vorstandsporträts, Werksaufnahmen, Produktbilder, Mitarbeiterfotos,
Anlagen, Stimmungsbilder.

Diese Klasse ist wichtiger als sie wirkt: Geschäftsberichte bestehen visuell zu
einem erheblichen Teil aus Fotografie, und das Modell muss lernen, diese sicher
abzulehnen.

### `other` — Sonstiges (inkl. Logos und Piktogramme)

Auffangklasse. Nicht optional: ohne sie liefert das Modell selbstbewusste
Falschaussagen für alles, was es nicht kennt.

**Typisch:**
- Konzern- und Markenlogos, Zertifizierungssiegel (ISO, FSC), Auszeichnungen,
  Ratingsymbole, Icon-Sets in Nachhaltigkeitskapiteln, SDG-Kacheln, Piktogramme
- Zeitstrahlen, Meilensteine, Roadmaps
- Wesentlichkeits- und Risikomatrizen
- KPI-Kacheln (große Zahl + Label + Pfeil)
- Fortschrittsbalken, Tachos, Ampeln
- Sankey, Radar, Tornado
- **Radial- und Nabe-Speiche-Diagramme** — ein zentrales Element mit ringsum
  angeordneten Elementen, mit oder ohne gezeichnete Speichen
- **Ring-, Rad- und Kreissegment-Grafiken**, bei denen der Kreis selbst die
  Grafik ist (Strategie-Räder, Kreisläufe mit aneinandergrenzenden Segmenten)
- **Prozesskreisläufe** — Kästen im Kreis mit Pfeilen (`flow` ist nie rund)
- **Gestapelte oder geschichtete Kastendiagramme ohne Verbindungen** —
  Governance-/Compliance-Strukturen als gestapelte Farbbänder, geschichtete
  Tempel-/Framework-Diagramme, Pyramiden
- Dekorative Grafik, Trennlinien, Extraktionsartefakte, leere Ausschnitte

**Abgrenzung:** Ein Icon oder Logo **innerhalb** eines Diagramms macht das Bild
nicht zu `other` — es zählt der Bildinhalt als Ganzes.

Bubble-Charts gehören zu `scatter` — nicht mehr hierher. (Logos und
Piktogramme, früher eigene Klasse `logo_icon`, gehören jetzt hierher.)
Ablaufdiagramme gehören **nur dann** zu `flow`, wenn gezeichnete Verbindungen
einen Pfad oder eine Hierarchie bilden — siehe Abschnitt `flow`; Radial-, Ring-
und verbindungslose Kastendiagramme bleiben hier.

Wenn eine dieser Unterkategorien häufig auftritt, im Notizfeld vermerken. Ab
einem gewissen Volumen lohnt sich eine eigene Klasse (siehe Abschnitt 5).

---

## 4. Sonderregeln

### R1 — Mehrere Diagramme verschiedener Typen in einem Bild

Ein extrahierter Ausschnitt enthält mehrere Diagramme **verschiedener Typen**
nebeneinander (in Kennzahlenkapiteln häufig) — z. B. ein Ringdiagramm neben
einem Balkendiagramm. Kein einzelnes Label ist korrekt, und die Diagramme
laufen in verschiedene nachgelagerte Parser.

**Regel:** Bild in die Nachbearbeitung geben und in Einzeldiagramme
zuschneiden. Jeder Zuschnitt bekommt sein eigenes Label.

**Wenn ein Zuschnitt nicht möglich ist** (Diagramme überlappen, gemeinsame
Legende, gemeinsame Achse): `other` labeln und im Notizfeld `multi` vermerken.

**Nicht als Mehrfachdiagramm gilt — als einzelne Diagrammklasse labeln,
Ausschnitt ganz lassen:**
- Mehrere Diagramme **desselben Typs** (z. B. zwei Ringe, drei
  Balkendiagramme). Das Label ist eindeutig (alle Ringe → `pie_donut`), also
  den Ausschnitt ganz lassen und nach diesem Typ labeln. Das Zerlegen in
  einzelne Instanzen ist Aufgabe des Parsers, nicht des Klassifikators — und
  der Ausschnitt erreicht den Klassifikator zur Inferenz als Ganzes (davor
  läuft kein Splitter), er muss also als Ganzes trainiert und validiert werden,
  sonst baut man einen Train/Inferenz-Versatz ein.
- Ein Plot mit mehreren Datenreihen.
- Ein Diagramm mit separater Legende.

**Prinzip:** Nach Verarbeitungsziel trennen, nicht nach Diagramm-Instanz. Alle
Diagramme laufen in dieselbe Klasse → ein Label, Ausschnitt ganz lassen. Sie
laufen in verschiedene Klassen → zuschneiden (oder `other` + `multi`, wenn nicht
sauber schneidbar).

### R2 — Infografik-Rahmen

Ein Diagramm ist in ein gestaltetes Layout eingebettet: Überschrift, Fließtext,
Icons, Farbflächen ringsum.

**Regel:** Nimmt das Diagramm **mehr als etwa die Hälfte** der Bildfläche ein →
Diagrammklasse. Sonst → `other`.

Diese Regel ist bewusst grob. Wichtig ist nicht, wo genau die Grenze liegt,
sondern dass alle dieselbe anwenden.

### R3 — `waterfall` gegen `bar_stacked`

Die schwierigste Unterscheidung; hier entstehen die meisten Fehler.

| Merkmal | `waterfall` | `bar_stacked` |
|---|---|---|
| Grundlinie | Nur erster und letzter Balken | Alle Balken |
| Segmente | Ein Segment je Balken, schwebend | Mehrere Segmente je Balken, aufeinander |
| Farben | Nach Richtung (Zu-/Abnahme) | Nach Kategorie |
| X-Achse | Veränderungsgründe | Jahre oder Kategorien |
| Verbindungslinien | Oft vorhanden | Nie |
| Vorzeichen in Labels | Häufig | Nie |

**Merksatz:** Eine Brücke erzählt, *wie* man von A nach B kommt. Ein Stapel
zeigt, *woraus* ein Wert besteht.

Bei echter Unsicherheit: Bild in `_unsure` legen und im Review besprechen.
Nicht raten — falsche Labels in genau diesem Klassenpaar richten den größten
Schaden an.

### R4 — Achsenlose Diagramme

Corporate Design streicht die Y-Achse häufig komplett und schreibt Werte direkt
an die Balken.

**Regel:** Solche Bilder werden ganz normal nach Diagrammtyp gelabelt. Fehlende
Achsen sind **kein** Grund für `other`.

Diese Fälle müssen im Trainingsset gut vertreten sein, sonst lernt das Modell
Achsengeometrie als Merkmal — die in echten Geschäftsberichten oft fehlt.

### R5 — Unsicherheit

Es gibt einen Ordner `_unsure`. Wer nach 15 Sekunden keine Entscheidung hat,
legt das Bild dort ab und geht weiter. Diese Bilder werden gesammelt im Review
entschieden.

`_unsure` ist keine Klasse und geht nicht ins Training. Ein wachsender
`_unsure`-Ordner ist ein Signal, dass dieser Guide eine Regel braucht.

### R6 — Schlechte Ausschnitte

Abgeschnittene, unscharfe, leere oder verrutschte Extraktionen kommen nach
`_broken`. Nicht ins Training, aber aufheben: sie zeigen Fehler in der
Extraktionspipeline.

---

## 5. Wann eine neue Klasse entsteht

Eine Klasse ist gerechtfertigt, wenn **beide** Bedingungen erfüllt sind:

1. **Nachgelagerter Nutzen:** Ein Verarbeitungsschritt behandelt sie anders als
   die bestehenden Klassen. Genau deshalb wurden `bar_vertical` und
   `bar_horizontal` zu `bar` zusammengelegt — sie werden nachgelagert identisch
   verarbeitet, die Trennung war überflüssig. `bar_grouped` und `bar_stacked`
   bleiben dagegen getrennt, weil beide schwerer zu parsen sind als ein
   einfacher Balken einer Reihe.
2. **Ausreichendes Volumen:** Mindestens rund 100 echte Beispiele sind
   auffindbar. Darunter lernt das Modell die Klasse nicht, verwässert aber die
   Nachbarklassen. Ausnahme: Für Klassen, die das Basismodell bereits kennt
   (`scatter`, `flow`), genügen weniger echte Beispiele, weil das Modell die
   Repräsentation nur schärft statt sie neu zu lernen — ein belastbares
   Validierungsset bleibt trotzdem Pflicht.

Kandidaten aus der `other`-Notizspalte, sobald das Volumen reicht:
`timeline`, `matrix`, `kpi_tile`, `area`, `gauge_progress`.

---

## 6. Arbeitsweise

**Ordnerstruktur.** Das Vorklassifikationsskript legt die Ausschnitte bereits in
Ordner mit vorgeschlagenen Labels. Korrigieren heißt: Datei in den richtigen
Ordner verschieben. Nicht verschieben heißt: Vorschlag bestätigt.

```
review/
├── bar/
├── bar_grouped/
├── bar_stacked/
├── waterfall/
├── line/
├── combo_bar_line/
├── scatter/
├── pie_donut/
├── flow/
├── map/
├── table/
├── photo/
├── other/
├── _unsure/      → nicht ins Training
└── _broken/      → nicht ins Training
```

**Reihenfolge.** Erst `_review` (niedrige Modellkonfidenz) durcharbeiten, dann
die übrigen Ordner gegenprüfen. Dort steckt der größte Informationsgewinn pro
Minute.

**Vorsicht vor Bestätigungsdrift.** Vorgeschlagene Labels sind bequem und
verführen zum Durchwinken. Besonders `waterfall` wird vom Basismodell
systematisch als `bar_chart` vorgeschlagen — dieser Ordner braucht die
gründlichste Durchsicht.

**Validierungsset.** Mindestens 50 bis 100 **echte** Bilder je Klasse, von Hand
gelabelt, niemals synthetisch, niemals aus dem Trainingsset. Ohne das ist jede
gemessene Genauigkeit wertlos.

**Qualitätssicherung.** Etwa 10 % der Bilder von einer zweiten Person unabhängig
labeln lassen. Übereinstimmung unter etwa 90 % bedeutet: dieser Guide ist an
einer Stelle unklar — nachschärfen, nicht die Labeler ermahnen.

---

## 7. Änderungshistorie

| Version | Datum | Änderung |
|---|---|---|
| 1.0 | — | Erstfassung: 12 Tier-1-Klassen, Regeln R1–R6 |
| 1.1 | 2026-08-23 | `combo_bar_line` geschärft: Kriterium ist, ob sich die Linie über die Kategorien verändert, nicht ob sie einen Legendeneintrag hat. Behebt den Widerspruch zur Zielwertlinien-Regel. |
| 1.2 | 2026-08-26 | `bar_vertical` + `bar_horizontal` zu `bar` zusammengelegt (nachgelagert identisch geparst). Gruppierte Balken als eigene Klasse `bar_grouped` ausgelagert, `bar_stacked` bleibt getrennt — beide schwerer zu parsen. Neue Klassen `scatter` (Abgrenzung gegen `line`) und `flow` (Prozess-/Ablauf-/Organigramme, späteres Parsing-Ziel). Jetzt 14 Tier-1-Klassen. |
| 1.3 | 2026-08-31 | R1 auf Diagramme **verschiedener** Typen eingegrenzt. Mehrere Diagramme **desselben** Typs (z. B. zwei Ringe) werden jetzt als diese eine Klasse gelabelt und ganz gelassen — das Label ist eindeutig und der Ausschnitt erreicht den Klassifikator zur Inferenz als Ganzes; das Zerlegen in Instanzen ist Aufgabe des Parsers. Entscheidungsbaum-Schritt 3 entsprechend umformuliert. |
| 1.4 | 2026-09-09 | `logo_icon` in `other` aufgegangen. Jetzt 13 Tier-1-Klassen. |
| 1.5 | 2026-09-12 | **`flow` verschärft — Breaking Change, Relabeling nötig.** `flow` erfordert jetzt explizite gezeichnete Struktur (Verbindungen zwischen den Elementen oder richtungsgebende Formen wie Chevrons und Trichterstufen) *und* Richtung/Hierarchie *und* ein nicht-kreisförmiges Layout. Nach `other` verschoben: Radial- und Nabe-Speiche-Diagramme (auch mit gezeichneten Speichen), Ring-/Rad-/Kreissegment-Grafiken, **Prozesskreisläufe** sowie gestapelte oder geschichtete Kastendiagramme ohne Verbindungen (Governance-Farbbänder, Tempel-/Framework-Schichten, Pyramiden). **Das kehrt die bisherige Regel um**, nach der allein durch das Layout ausgedrückte Hierarchie als Verbindung zählte. „`flow` ist nie rund" gilt jetzt absolut und ausnahmslos, was dem Klassifikator zugleich ein sauberes Unterscheidungsmerkmal gibt. Anlass: auf dem Test-Set lag der `flow`-Recall bei 42,5 %, 50,9 % der `flow`-Bilder gingen nach `other`, umgekehrt nur 1,6 % — und dieselben visuellen Familien (Governance-Räder, Organigramm-Farbbänder) fanden sich auf beiden Seiten gelabelt, die alte Definition war also nicht konsistent anwendbar. |
