"""Rapport HTML : vos tendances, vos priorités et le détail de vos dernières parties.

    python -m src.report --raw-dir data/raw/me --out results/report.html

Un seul fichier, sans dépendance ni ressource externe : il s'ouvre dans un navigateur et se garde tel quel.
Il ne contient pas votre puuid, mais il contient vos identifiants de parties : à partager en connaissance de cause.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import pandas as pd

from .benchmarks import DEFAULT_BENCH, ROLE_NAMES, STAT_COLS, load_benchmarks
from .debrief import LABELS, STAT_FORMAT, _format, _fr
from .progress import champion_table, main_role, player_history, priorities, trend_table
from .winprob import EARLY_MINUTE, format_gaps

LEGAL = ("lol-analyzer isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone "
         "officially involved in producing or managing Riot Games properties. Riot Games, and all associated "
         "properties are trademarks or registered trademarks of Riot Games, Inc.")
WIDTH, HEIGHT = 720, 340
LEFT, RIGHT, TOP, BOTTOM = 52, 42, 18, 36
TAG_SPACING = 64  # en dessous, deux étiquettes de moments clés voisines se chevaucheraient
VERDICT_MARKS = {"en progrès": ("▲", "good"), "en baisse": ("▼", "bad"), "pas de conclusion": ("=", "neutral"),
                 "pas assez de parties": ("·", "neutral")}

STYLE = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --series: #2a78d6; --wash: #f0efec; --line: rgba(11,11,11,.10);
  --good: #006300; --bad: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --series: #3987e5; --wash: #262624; --line: rgba(255,255,255,.10);
    --good: #0ca30c; --bad: #e66767;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); line-height: 1.5;
       font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 32px 16px 48px; }
h1 { font-size: 28px; line-height: 1.2; margin: 0 0 4px; font-weight: 650; }
h2 { font-size: 20px; margin: 40px 0 4px; font-weight: 650; }
h4 { font-size: 15px; margin: 16px 0 4px; font-weight: 650; }
p { margin: 0 0 8px; }
.lead, .note { color: var(--ink-2); }
.note { font-size: 14px; }
code { font: 13px/1.4 ui-monospace, Consolas, monospace; background: var(--wash); padding: 2px 4px;
       border-radius: 4px; overflow-wrap: anywhere; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px; margin-top: 12px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 20px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 16px; }
.tile .label { color: var(--ink-2); font-size: 14px; }
.tile .value { font-size: 26px; font-weight: 650; line-height: 1.3; overflow-wrap: anywhere; }
.tile .detail { color: var(--ink-2); font-size: 13px; }
ul { margin: 0; padding-left: 20px; }
li { margin-bottom: 6px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th { text-align: left; font-weight: 600; color: var(--ink-2); }
th, td { padding: 6px 12px 6px 0; border-bottom: 1px solid var(--grid); white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.mark { display: inline-block; width: 1.1em; text-align: center; font-weight: 700; }
.good { color: var(--good); } .bad { color: var(--bad); } .neutral { color: var(--muted); }
.bars { display: grid; gap: 8px; margin: 0; padding: 0; list-style: none; }
.bars li { display: grid; grid-template-columns: minmax(120px, 210px) 1fr 3.2em; gap: 12px; align-items: center;
           margin: 0; border-radius: 4px; }
.bars li:hover .fill, .bars li:focus .fill { opacity: .8; }
.bars li:focus-visible { outline: 2px solid var(--series); outline-offset: 2px; }
.bars .name { font-size: 14px; color: var(--ink-2); }
.track { position: relative; display: block; height: 14px; border-bottom: 1px solid var(--axis); }
.fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--series); border-radius: 0 4px 4px 0; }
.middle { position: absolute; left: 50%; top: -3px; bottom: -1px; width: 1px; background: var(--muted); }
.bars .value { font-size: 14px; text-align: right; font-variant-numeric: tabular-nums; }
details.game { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; margin-top: 12px; }
details.game > summary { padding: 12px 16px; cursor: pointer; font-weight: 600; overflow-wrap: anywhere; }
details.game > summary .meta { font-weight: 400; color: var(--ink-2); font-size: 14px; }
details.game > div { padding: 0 16px 16px; }
details.values { margin-top: 12px; }
details.values summary { cursor: pointer; font-size: 14px; color: var(--ink-2); }
svg.curve { display: block; width: 100%; height: auto; touch-action: pan-y; border-radius: 4px; }
svg.curve:focus-visible { outline: 2px solid var(--series); outline-offset: 2px; }
svg.curve text { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--muted); }
svg.curve text.tag { fill: var(--ink); font-weight: 650; }
svg.curve .grid { stroke: var(--grid); stroke-width: 1; }
svg.curve .even { stroke: var(--axis); stroke-width: 1; }
svg.curve .early { fill: var(--wash); }
svg.curve .series { fill: none; stroke: var(--series); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
svg.curve .dot { fill: var(--series); stroke: var(--surface); stroke-width: 2; }
svg.curve .cross { stroke: var(--muted); stroke-width: 1; }
.tip { position: fixed; z-index: 10; pointer-events: none; background: var(--surface); color: var(--ink);
       border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; font-size: 13px; max-width: 280px;
       box-shadow: 0 2px 8px rgba(0,0,0,.15); }
.tip strong { display: block; font-size: 15px; }
footer { margin-top: 48px; color: var(--muted); font-size: 12px; }
@media (max-width: 560px) {
  .bars li { grid-template-columns: 1fr 3.2em; gap: 2px 12px; }
  .bars .name { grid-column: 1 / -1; }
  /* la courbe est réduite à moins de la moitié de sa taille : ses traits et son texte doivent le compenser */
  svg.curve text { font-size: 24px; }
  svg.curve .unit { display: none; }
  svg.curve .series { stroke-width: 4; }
  svg.curve .grid, svg.curve .even, svg.curve .cross { stroke-width: 2; }
  svg.curve .dot { r: 9px; stroke-width: 4; }
}
"""

SCRIPT = """
(function () {
  var tip = document.getElementById('tip');
  function showTip(value, label, x, y) {
    tip.textContent = '';
    var strong = document.createElement('strong');
    strong.textContent = value;
    tip.appendChild(strong);
    tip.appendChild(document.createTextNode(label));
    tip.hidden = false;
    var box = tip.getBoundingClientRect();
    tip.style.left = Math.max(8, Math.min(x + 12, window.innerWidth - box.width - 8)) + 'px';
    tip.style.top = Math.max(8, y - box.height - 12) + 'px';
  }
  function hideTip() { tip.hidden = true; }

  document.querySelectorAll('svg.curve').forEach(function (svg) {
    var points = JSON.parse(svg.dataset.points);
    var cross = svg.querySelector('.cross');
    var marker = svg.querySelector('.hover');
    var width = svg.viewBox.baseVal.width;
    var left = +svg.dataset.left, plot = +svg.dataset.plot, top = +svg.dataset.top, tall = +svg.dataset.tall;
    var first = points[0][0], span = (points[points.length - 1][0] - first) || 1;
    var current = points.length - 1;
    function px(minute) { return left + (minute - first) / span * plot; }
    function py(pct) { return top + (1 - pct / 100) * tall; }
    function select(i, clientX, clientY) {
      current = Math.max(0, Math.min(points.length - 1, i));
      var p = points[current], x = px(p[0]), y = py(p[1]);
      cross.setAttribute('x1', x); cross.setAttribute('x2', x); cross.hidden = false; cross.style.display = '';
      marker.setAttribute('cx', x); marker.setAttribute('cy', y); marker.style.display = '';
      var box = svg.getBoundingClientRect(), scale = box.width / width;
      showTip(p[1] + ' %', p[0] + 'e minute', clientX === undefined ? box.left + x * scale : clientX,
              clientY === undefined ? box.top + y * scale : clientY);
    }
    function clear() { cross.style.display = 'none'; marker.style.display = 'none'; hideTip(); }
    svg.addEventListener('pointermove', function (event) {
      var box = svg.getBoundingClientRect();
      var minute = first + ((event.clientX - box.left) * width / box.width - left) / plot * span;
      var best = 0;
      points.forEach(function (p, i) {
        if (Math.abs(p[0] - minute) < Math.abs(points[best][0] - minute)) { best = i; }
      });
      select(best, event.clientX, event.clientY);
    });
    svg.addEventListener('pointerleave', clear);
    svg.addEventListener('focus', function () { select(current); });
    svg.addEventListener('blur', clear);
    svg.addEventListener('keydown', function (event) {
      if (event.key === 'ArrowLeft') { select(current - 1); event.preventDefault(); }
      if (event.key === 'ArrowRight') { select(current + 1); event.preventDefault(); }
    });
    clear();
  });

  document.querySelectorAll('[data-value]').forEach(function (row) {
    function show(event) {
      var box = row.getBoundingClientRect();
      showTip(row.dataset.value, row.dataset.label, event.clientX || box.left + box.width / 2, event.clientY || box.top);
    }
    row.addEventListener('pointermove', show);
    row.addEventListener('pointerleave', hideTip);
    row.addEventListener('focus', show);
    row.addEventListener('blur', hideTip);
  });
})();
"""


def curve_svg(analysis: dict) -> str:
    """Courbe de probabilité de victoire. Dessinée côté serveur : elle s'affiche même sans JavaScript."""
    points = [[c["minute"], round(c["win_prob"] * 100)] for c in analysis["curve"]]
    first, last = points[0][0], points[-1][0]
    plot, tall = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def x(minute: float) -> float:
        return round(LEFT + (minute - first) / ((last - first) or 1) * plot, 1)

    def y(pct: float) -> float:
        return round(TOP + (1 - pct / 100) * tall, 1)

    parts = []
    if first < EARLY_MINUTE:
        edge = x(min(EARLY_MINUTE, last))
        parts.append(f'<rect class="early" x="{x(first)}" y="{TOP}" width="{edge - x(first)}" height="{tall}"/>')
        parts.append(f'<text x="{x(first) + 4}" y="{HEIGHT - BOTTOM - 6}">peu fiable</text>')
    for pct in (0, 25, 50, 75, 100):
        parts.append(f'<line class="{"even" if pct == 50 else "grid"}" x1="{LEFT}" x2="{WIDTH - RIGHT}" '
                     f'y1="{y(pct)}" y2="{y(pct)}"/>')
        parts.append(f'<text x="{LEFT - 8}" y="{y(pct) + 4}" text-anchor="end">{pct}<tspan class="unit"> %</tspan></text>')
    for minute in range(first + (-first % 5), last + 1, 5):
        parts.append(f'<text x="{x(minute)}" y="{HEIGHT - 12}" text-anchor="middle">{minute}</text>')
    path = " ".join(f"{'L' if i else 'M'}{x(m)} {y(pct)}" for i, (m, pct) in enumerate(points))
    parts.append(f'<path class="series" d="{path}"/>')
    previous = None
    for tp in analysis["turning_points"]:
        minute, pct = tp["minute"], round(tp["win_prob_after"] * 100)
        crowded = previous is not None and x(minute) - previous < TAG_SPACING
        below = pct > 12 and (pct > 85 or crowded)
        parts.append(f'<circle class="dot" cx="{x(minute)}" cy="{y(pct)}" r="5"/>')
        parts.append(f'<text class="tag" x="{x(minute)}" y="{y(pct) + 30 if below else y(pct) - 12}" '
                     f'text-anchor="middle">{minute}e</text>')
        previous = x(minute)
    parts.append(f'<circle class="dot" cx="{x(last)}" cy="{y(points[-1][1])}" r="4"/>')
    parts.append(f'<text class="tag" x="{x(last) + 8}" y="{y(points[-1][1]) + 4}">{points[-1][1]}'
                 '<tspan class="unit"> %</tspan></text>')
    parts.append(f'<line class="cross" x1="0" x2="0" y1="{TOP}" y2="{TOP + tall}" style="display:none"/>')
    parts.append('<circle class="dot hover" r="5" style="display:none"/>')
    moments = ", ".join(f"{tp['minute']}e" for tp in analysis["turning_points"]) or "aucun"
    label = (f"Probabilité de victoire de l'équipe, de {points[0][1]} % à la minute {first} à {points[-1][1]} % à la "
             f"minute {last}. Moments clés : {moments}. Flèches gauche et droite pour parcourir les valeurs.")
    return (f'<svg class="curve" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" tabindex="0" aria-label="{escape(label)}" '
            f'data-points="{json.dumps(points)}" data-left="{LEFT}" data-plot="{plot}" data-top="{TOP}" '
            f'data-tall="{tall}">{"".join(parts)}</svg>')


def _values_table(analysis: dict) -> str:
    rows = "".join(f'<tr><td class="num">{c["minute"]}</td><td class="num">{round(c["win_prob"] * 100)} %</td></tr>'
                   for c in analysis["curve"])
    return ('<details class="values"><summary>Toutes les valeurs de la courbe</summary><div class="scroll"><table>'
            f'<tr><th class="num">Minute</th><th class="num">Probabilité</th></tr>{rows}</table></div></details>')


def debrief_html(markdown: str) -> str:
    """Le débrief mis en page : titres, puces et paragraphes, sans la vérification ni les faits qui le suivent."""
    out: list[str] = []
    in_list = False
    for raw in markdown.split("\n---\n", 1)[0].splitlines():
        line = raw.strip()
        is_item = line.startswith("- ")
        if in_list and not is_item:
            out.append("</ul>")
            in_list = False
        if not line or line.startswith("# "):
            continue
        title = re.fullmatch(r"\*\*(.+)\*\*", line)
        if title:
            out.append(f"<h4>{escape(title.group(1))}</h4>")
        elif is_item:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{escape(line[2:])}</li>")
        else:
            out.append(f"<p>{escape(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def _moments_html(analysis: dict) -> str:
    """Les moments clés quand aucun débrief n'a été écrit."""
    items = []
    for tp in analysis["turning_points"]:
        text = (f"{tp['minute']}e minute : la probabilité de l'équipe passe de {round(tp['win_prob_before'] * 100)} % à "
                f"{round(tp['win_prob_after'] * 100)} %.")
        if tp.get("events"):
            text += " Événements : " + " ; ".join(tp["events"]) + "."
        if tp.get("ecarts"):
            text += f" Écarts sur la minute : {format_gaps(tp['ecarts'])}."
        items.append(f"<li>{escape(text)}</li>")
    return f"<h4>Moments clés</h4><ul>{''.join(items)}</ul>" if items else ""


def _find_debrief(debriefs_dir: Path, match_id: str) -> str | None:
    for suffix in ("", "_gabarit"):
        path = debriefs_dir / f"{match_id}{suffix}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


def _game_html(analysis: dict, created_ms: int, debriefs_dir: Path, is_open: bool) -> str:
    who = analysis["perspective"]
    role = ROLE_NAMES.get(who.get("role"), who.get("role") or "poste inconnu")
    when = datetime.fromtimestamp(created_ms / 1000, tz=UTC).strftime("%d/%m/%Y")
    debrief = _find_debrief(debriefs_dir, analysis["match_id"])
    details = debrief_html(debrief) if debrief else _moments_html(analysis)
    return (f'<details class="game"{" open" if is_open else ""}><summary>{escape(str(who.get("champion") or "Équipe " + who["team"]))} '
            f'({escape(str(role))}), {escape(who["result"])} <span class="meta">{when} · {escape(analysis["match_id"])}</span>'
            f"</summary><div>{curve_svg(analysis)}{details}{_values_table(analysis)}</div></details>")


def _games_html(raw_dir: Path, curves_dir: Path, debriefs_dir: Path, limit: int) -> str:
    games = []
    for path in curves_dir.glob("*.json"):
        match_path = raw_dir / "matches" / path.name
        analysis = json.loads(path.read_text(encoding="utf-8"))
        if match_path.exists() and analysis.get("curve"):
            created = json.loads(match_path.read_text(encoding="utf-8"))["info"].get("gameCreation", 0)
            games.append((created, analysis))
    games = sorted(games, key=lambda g: g[0], reverse=True)[:limit]
    return "".join(_game_html(analysis, created, debriefs_dir, i == 0) for i, (created, analysis) in enumerate(games))


def _tiles_html(history: pd.DataFrame, role: str | None, window: int) -> str:
    recent = history.tail(window)
    top = champion_table(history).iloc[0]
    tiles = [
        ("Parties suivies", str(len(history)), "de plus de 15 minutes"),
        ("Victoires récentes", f"{round(100 * recent['win'].mean())} %", f"sur vos {len(recent)} dernières parties"),
        ("Poste principal", ROLE_NAMES.get(role, str(role)), f"sur vos {min(len(history), 2 * window)} dernières parties"),
        ("Champion le plus joué", str(top["champion"]), f"{top['parties']} parties, {top['victoires_pct']} % de victoires"),
    ]
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div>'
        f'<div class="detail">{escape(detail)}</div></div>' for label, value, detail in tiles) + "</div>"


def _position_html(rows: list[dict], role: str, window: int) -> str:
    bars = "".join(
        f'<li tabindex="0" data-value="{row["percentile"]} %" data-label="{escape(row["nom"])} : {escape(row["valeur"])}, '
        f'médiane {escape(row["mediane"])}"><span class="name">{escape(row["nom"])}</span>'
        f'<span class="track"><span class="fill" style="width:{row["percentile"]}%"></span><span class="middle"></span></span>'
        f'<span class="value">{row["percentile"]} %</span></li>' for row in rows)
    return (f"<h2>Votre position</h2><p class=\"lead\">Part des {escape(rows[0]['reference'])} que votre moyenne dépasse, "
            f"sur vos {window} dernières parties au poste {escape(ROLE_NAMES.get(role, role))}. Le trait marque la "
            f"médiane.</p><div class=\"card\"><ul class=\"bars\">{bars}</ul></div>"
            "<p class=\"note\" style=\"margin-top:8px\">Une moyenne varie moins qu'une partie : l'écart réel avec les "
            "autres joueurs est plus marqué que ce percentile.</p>")


def _stat_text(stat: str, value: float, signed: bool = False) -> str:
    if pd.isna(value):
        return "–"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{_fr(_format(stat, value))}{LABELS[STAT_FORMAT[stat][0]][1]}"


def _trend_html(table: pd.DataFrame, window: int) -> str:
    rows = []
    for r in table.itertuples():
        symbol, tone = VERDICT_MARKS[r.verdict]
        rows.append(f'<tr><td>{escape(LABELS[STAT_FORMAT[r.stat][0]][0])}</td>'
                    f'<td class="num">{_stat_text(r.stat, r.precedent)}</td><td class="num">{_stat_text(r.stat, r.recent)}</td>'
                    f'<td class="num">{_stat_text(r.stat, r.evolution, signed=True)}</td>'
                    f'<td><span class="mark {tone}">{symbol}</span> {escape(r.verdict)}</td></tr>')
    return (f'<h2>Ce qui évolue</h2><p class="lead">Moyennes de vos {window} dernières parties au poste principal, contre '
            f'les {window} précédentes. « Pas de conclusion » : l\'écart reste dans le bruit normal d\'une partie à '
            'l\'autre.</p><div class="card scroll"><table><tr><th>Statistique</th><th class="num">Précédentes</th>'
            f'<th class="num">Récentes</th><th class="num">Évolution</th><th>Verdict</th></tr>{"".join(rows)}</table></div>')


def _champions_html(table: pd.DataFrame) -> str:
    rows = "".join(f'<tr><td>{escape(str(r.champion))}</td><td class="num">{r.parties}</td>'
                   f'<td class="num">{r.victoires_pct} %</td><td class="num">{_fr(r.kda)}</td></tr>'
                   for r in table.head(10).itertuples())
    return ('<h2>Vos champions</h2><p class="lead">Sur peu de parties, un taux de victoire ne veut pas dire grand-chose.</p>'
            '<div class="card scroll"><table><tr><th>Champion</th><th class="num">Parties</th><th class="num">Victoires</th>'
            f'<th class="num">KDA moyen</th></tr>{rows}</table></div>')


def build_report(raw_dir: Path, curves_dir: Path, debriefs_dir: Path, bench: pd.DataFrame | None,
                 limit: int = 20, window: int = 20) -> str:
    puuid_file, riot_id_file = raw_dir / "puuid.txt", raw_dir / "riot_id.txt"
    puuid = puuid_file.read_text(encoding="utf-8").strip() if puuid_file.exists() else None
    riot_id = riot_id_file.read_text(encoding="utf-8").strip() if riot_id_file.exists() else None
    history = player_history(raw_dir, puuid)
    title = f"Rapport de {riot_id}" if riot_id else "Rapport de vos parties"

    sections = [f"<h1>{escape(title)}</h1>"]
    if history.empty:
        sections.append('<p class="lead">Aucune partie pour l\'instant. Lancez '
                        '<code>python -m src.run report --riot-id "Pseudo#EUW"</code>.</p>')
    else:
        role = main_role(history, window)
        rows = priorities(history, bench, role, window, k=len(STAT_COLS))
        sections.append(_tiles_html(history, role, window))
        if rows:
            sections.append("<h2>À travailler en priorité</h2><div class=\"card\"><ul>"
                            + "".join(f"<li>{escape(row['phrase'])}</li>" for row in rows[:2]) + "</ul></div>")
            sections.append(_position_html(rows, role, min(window, int((history["role"] == role).sum()))))
        else:
            sections.append('<p class="note" style="margin-top:16px">Pas de repères pour ce poste : lancez '
                            '<code>python -m src.run prepare</code> pour vous comparer à d\'autres joueurs.</p>')
        sections.append(_trend_html(trend_table(history, window, role), window))
        sections.append(_champions_html(champion_table(history)))

    games = _games_html(raw_dir, curves_dir, debriefs_dir, limit)
    if games:
        sections.append('<h2>Vos dernières parties</h2><p class="lead">Probabilité de victoire de votre équipe, minute par '
                        "minute. Un saut coïncide avec des événements : il n'en prouve pas la cause.</p>" + games)
    elif not history.empty:
        sections.append('<p class="note" style="margin-top:16px">Pas de courbes de parties : lancez '
                        '<code>python -m src.run prepare</code> pour entraîner le modèle.</p>')
    today = datetime.now(UTC).strftime("%d/%m/%Y")
    return ('<!doctype html><html lang="fr"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{escape(title)}</title><style>{STYLE}</style></head><body><main>{''.join(sections)}"
            f"<footer><p>Généré le {today} par lol-analyzer.</p><p>{escape(LEGAL, quote=False)}</p></footer></main>"
            f'<div class="tip" id="tip" hidden></div><script>{SCRIPT}</script></body></html>')


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw/me")
    p.add_argument("--curves", default="results/curves")
    p.add_argument("--debriefs", default="results/debriefs")
    p.add_argument("--bench", default=str(DEFAULT_BENCH))
    p.add_argument("--limit", type=int, default=20, help="nombre de parties détaillées")
    p.add_argument("--window", type=int, default=20, help="nombre de parties récentes pour les tendances")
    p.add_argument("--out", default="results/report.html")
    args = p.parse_args(argv)

    html = build_report(Path(args.raw_dir), Path(args.curves), Path(args.debriefs), load_benchmarks(args.bench),
                        args.limit, args.window)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Rapport écrit : {out}")


if __name__ == "__main__":
    main()
