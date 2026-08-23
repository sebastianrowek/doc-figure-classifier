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
| `bar_vertical` | Säulendiagramm | Senkrechte Balken auf gemeinsamer Grundlinie |
| `bar_horizontal` | Balkendiagramm | Waagerechte Balken auf gemeinsamer Grundlinie |
| `bar_stacked` | Gestapelte Säulen/Balken | Segmente innerhalb eines Balkens aufeinandergesetzt |
| `waterfall` | Wasserfall / Brücke | Schwebende Balken zwischen Start- und Endwert |
| `line` | Liniendiagramm | Werte als verbundener Linienzug |
| `combo_bar_line` | Kombidiagramm | Balken **und** Linie in einem Plot |
| `pie_donut` | Kreis-/Ringdiagramm | Kreisförmige Anteilsdarstellung |
| `map` | Karte | Geografische Darstellung |
| `table` | Tabelle | Zeilen-/Spaltenraster ohne grafische Kodierung |
| `photo` | Foto | Fotografische Aufnahme |
| `logo_icon` | Logo / Piktogramm | Marke, Siegel, Icon, Award |
| `other` | Sonstiges | Alles, was in keine Klasse oben passt |

---

## 2. Entscheidungsbaum

Von oben nach unten durchgehen, beim ersten Treffer stoppen.

```
1. Ist es eine Fotografie (Personen, Gebäude, Produkte)?      → photo
2. Ist es ein Logo, Siegel, Award oder Piktogramm?            → logo_icon
3. Zeigt das Bild MEHRERE eigenständige Diagramme?            → Regel R1 (Abschnitt 4)
4. Ist eine geografische Karte das dominante Element?         → map
5. Sind Balken UND eine Linie als Datenreihen vorhanden?      → combo_bar_line
6. Kreisförmige Anteilsdarstellung (Voll- oder Ringform)?     → pie_donut
7. Schweben Balken zwischen Start- und Endwert (Brücke)?      → waterfall
8. Sind Balken in Segmente unterteilt (gestapelt)?            → bar_stacked
9. Sind es einfache Balken?
      senkrecht → bar_vertical      waagerecht → bar_horizontal
10. Ist es ein Linienzug (auch Fläche darunter)?              → line
11. Reines Zeilen-/Spaltenraster ohne Grafik?                 → table
12. Sonst                                                     → other
```

Die Reihenfolge ist bewusst gewählt: Schritt 5 vor 7/8/9, weil ein Kombidiagramm
sonst je nach Blickwinkel als Balken- oder Liniendiagramm gelabelt würde.

---

## 3. Klassen im Detail

### `bar_vertical` — Säulendiagramm

Senkrechte Balken, alle auf derselben Grundlinie beginnend.

**Typisch:** Umsatz je Geschäftsjahr, EBIT-Entwicklung, Mitarbeiterzahl,
F&E-Aufwand, Investitionen.

**Gehört ebenfalls hierher:**
- Gruppierte Säulen (mehrere Reihen nebeneinander, z. B. Vorjahr/Berichtsjahr).
  Gruppierung ist eine Eigenschaft der Daten, kein eigener Diagrammtyp.
- Säulen mit Datenbeschriftung statt Y-Achse.
- Einzelne hervorgehobene Säule (Farbakzent).

**Nicht hierher:**
- Balken in Segmente unterteilt → `bar_stacked`
- Balken beginnen nicht auf der Grundlinie → `waterfall`
- Zusätzliche Linie als Datenreihe → `combo_bar_line`

### `bar_horizontal` — Balkendiagramm

Wie oben, nur waagerecht.

**Typisch:** Umsatz nach Region, Top-10-Rankings, Altersstruktur der Belegschaft,
Umfrageergebnisse.

**Abgrenzungshinweis:** Die Orientierung entscheidet, nicht die Semantik. Ein
Umsatz-nach-Region-Chart kann je nach Layout in `bar_vertical` oder
`bar_horizontal` fallen — beides ist korrekt.

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

### `table` — Tabelle

Reines Zeilen-/Spaltenraster ohne grafische Kodierung von Werten.

**Typisch:** Kennzahlenübersicht, Mehrjahresvergleich, die als Bild statt als
Text im PDF vorliegt.

**Nicht hierher:** Tabellen mit eingebetteten Balken, Ampeln oder Sparklines —
diese als `other` labeln und in der Notizspalte vermerken.

### `photo` — Fotografie

**Typisch:** Vorstandsporträts, Werksaufnahmen, Produktbilder, Mitarbeiterfotos,
Anlagen, Stimmungsbilder.

Diese Klasse ist wichtiger als sie wirkt: Geschäftsberichte bestehen visuell zu
einem erheblichen Teil aus Fotografie, und das Modell muss lernen, diese sicher
abzulehnen.

### `logo_icon` — Logo und Piktogramm

**Typisch:** Konzern- und Markenlogos, Zertifizierungssiegel (ISO, FSC),
Auszeichnungen, Ratingsymbole, Icon-Sets in Nachhaltigkeitskapiteln,
SDG-Kacheln.

**Abgrenzung:** Ein Icon **innerhalb** eines Diagramms macht das Bild nicht zu
`logo_icon` — es zählt der Bildinhalt als Ganzes.

### `other` — Sonstiges

Auffangklasse. Nicht optional: ohne sie liefert das Modell selbstbewusste
Falschaussagen für alles, was es nicht kennt.

**Typisch:**
- Organigramme, Prozess- und Ablaufdiagramme, Wertschöpfungsketten
- Zeitstrahlen, Meilensteine, Roadmaps
- Wesentlichkeits- und Risikomatrizen
- KPI-Kacheln (große Zahl + Label + Pfeil)
- Fortschrittsbalken, Tachos, Ampeln
- Sankey, Radar, Bubble, Tornado
- Dekorative Grafik, Trennlinien, Extraktionsartefakte, leere Ausschnitte

Wenn eine dieser Unterkategorien häufig auftritt, im Notizfeld vermerken. Ab
einem gewissen Volumen lohnt sich eine eigene Klasse (siehe Abschnitt 5).

---

## 4. Sonderregeln

### R1 — Mehrfachdiagramme in einem Bild

Ein extrahierter Ausschnitt enthält mehrere eigenständige Diagramme
nebeneinander (in Kennzahlenkapiteln häufig).

**Regel:** Bild in die Nachbearbeitung geben und in Einzeldiagramme
zuschneiden. Jeder Zuschnitt bekommt sein eigenes Label.

**Wenn ein Zuschnitt nicht möglich ist** (Diagramme überlappen, gemeinsame
Legende, gemeinsame Achse): `other` labeln und im Notizfeld `multi` vermerken.

**Nicht als Mehrfachdiagramm gilt:** ein Plot mit mehreren Datenreihen; ein
Diagramm mit separater Legende.

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
   die bestehenden Klassen. Wenn `bar_vertical` und `bar_horizontal`
   nachgelagert identisch verarbeitet werden, war die Trennung überflüssig.
2. **Ausreichendes Volumen:** Mindestens rund 100 echte Beispiele sind
   auffindbar. Darunter lernt das Modell die Klasse nicht, verwässert aber die
   Nachbarklassen.

Kandidaten aus der `other`-Notizspalte, sobald das Volumen reicht:
`org_chart`, `process_flow`, `timeline`, `matrix`, `kpi_tile`, `area`,
`gauge_progress`.

---

## 6. Arbeitsweise

**Ordnerstruktur.** Das Vorklassifikationsskript legt die Ausschnitte bereits in
Ordner mit vorgeschlagenen Labels. Korrigieren heißt: Datei in den richtigen
Ordner verschieben. Nicht verschieben heißt: Vorschlag bestätigt.

```
review/
├── bar_vertical/
├── bar_horizontal/
├── bar_stacked/
├── waterfall/
├── line/
├── combo_bar_line/
├── pie_donut/
├── map/
├── table/
├── photo/
├── logo_icon/
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
