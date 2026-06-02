import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Atom,
  Binary,
  BookmarkPlus,
  Database,
  Download,
  FlaskConical,
  GitCompare,
  Network,
  Play,
  Radar,
  Search,
  Sparkles,
} from "lucide-react";
import {
  buildIndex,
  compareMolecules,
  discoverMolecules,
  exportShortlist,
  getMapPoints,
  getStatus,
  searchMolecules,
  structureSvgUrl,
} from "./api";
import { ChemicalMap } from "./components/ChemicalMap";
import type { CompareResponse, IndexStatus, MapPoint, SearchFilters, SearchMode, SearchResponse, SearchResult } from "./types/molspace";

const defaultFilters: SearchFilters = {
  molecular_weight_min: null,
  molecular_weight_max: 500,
  logp_min: null,
  logp_max: 5,
  qed_min: 0.35,
  lipinski_max: 1,
  target_class: null,
  exclude_high_toxicity: true,
  different_target_class: false,
};

const modes: Array<{ id: SearchMode; label: string; icon: typeof Atom }> = [
  { id: "structure", label: "Structure", icon: Atom },
  { id: "bioactivity", label: "Bioactivity", icon: Activity },
  { id: "hybrid", label: "Hybrid", icon: Binary },
  { id: "analog", label: "Target Shift", icon: Network },
];

function App() {
  const [status, setStatus] = useState<IndexStatus | null>(null);
  const [query, setQuery] = useState("aspirin");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [filters, setFilters] = useState<SearchFilters>(defaultFilters);
  const [mapPoints, setMapPoints] = useState<MapPoint[]>([]);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | undefined>();
  const [positiveIds, setPositiveIds] = useState<string[]>([]);
  const [negativeIds, setNegativeIds] = useState<string[]>([]);
  const [shortlistIds, setShortlistIds] = useState<string[]>([]);
  const [heatmap, setHeatmap] = useState(true);
  const [compare, setCompare] = useState<CompareResponse | null>(null);
  const [compareLeft, setCompareLeft] = useState("CHEMBL25");
  const [compareRight, setCompareRight] = useState("CHEMBL521");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("Ready");

  const resultById = useMemo(() => new Map((response?.results ?? []).map((result) => [result.molecule.molecule_id, result])), [response]);
  const selectedResult = selectedId ? resultById.get(selectedId) : response?.results[0];
  const targetClasses = useMemo(() => Array.from(new Set(mapPoints.map((point) => point.target_class))).sort(), [mapPoints]);

  useEffect(() => {
    void refreshStatus();
  }, []);

  async function refreshStatus() {
    const current = await getStatus();
    setStatus(current);
    if (current.exists && current.count > 0) {
      const points = await getMapPoints();
      setMapPoints(points);
      if (!response) {
        await runSearch("hybrid", false);
      }
    }
  }

  async function runBuild() {
    setBusy(true);
    setMessage("Building Qdrant molecule index");
    try {
      await buildIndex(1000, true);
      const points = await getMapPoints();
      setMapPoints(points);
      await refreshStatus();
      await runSearch(mode, false);
      setMessage("Index ready");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Index build failed");
    } finally {
      setBusy(false);
    }
  }

  async function runSearch(nextMode = mode, userStarted = true) {
    setBusy(true);
    if (userStarted) setMessage(`Running ${nextMode} search`);
    try {
      const nextFilters = nextMode === "analog" ? { ...filters, different_target_class: true } : filters;
      const data = await searchMolecules(query, nextMode, nextFilters);
      setResponse(data);
      setMode(nextMode);
      setSelectedId(data.results[0]?.molecule.molecule_id);
      setMessage(`${data.results.length} candidates returned`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Search failed");
    } finally {
      setBusy(false);
    }
  }

  async function runDiscovery() {
    setBusy(true);
    setMessage("Steering discovery through Qdrant context");
    try {
      const data = await discoverMolecules(query, positiveIds, negativeIds, filters);
      setResponse(data);
      setSelectedId(data.results[0]?.molecule.molecule_id);
      setMessage(`${data.results.length} steered candidates returned`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Discovery failed");
    } finally {
      setBusy(false);
    }
  }

  async function runCompare(left = compareLeft, right = compareRight) {
    setBusy(true);
    try {
      const data = await compareMolecules(left, right);
      setCompare(data);
      setCompareLeft(left);
      setCompareRight(right);
      setMessage("Comparison ready");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Compare failed");
    } finally {
      setBusy(false);
    }
  }

  async function downloadShortlist(format: "json" | "csv") {
    if (!shortlistIds.length) return;
    const blob = await exportShortlist(shortlistIds, format);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `molecule-space-shortlist.${format}`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function addUnique(setter: (value: string[]) => void, values: string[], id?: string) {
    if (!id) return;
    setter(Array.from(new Set([...values, id])));
  }

  function removeId(setter: (value: string[]) => void, values: string[], id: string) {
    setter(values.filter((item) => item !== id));
  }

  return (
    <main className="app-shell">
      <aside className="left-panel">
        <div className="brand-row">
          <div className="brand-mark">
            <img src="/molecule-space-logo.svg" alt="" />
          </div>
          <div>
            <h1>Molecule Space</h1>
            <p>Qdrant molecular discovery navigator</p>
          </div>
        </div>

        <section className="control-section">
          <label htmlFor="query">Molecule</label>
          <div className="search-row">
            <Search size={18} />
            <input id="query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="aspirin or SMILES" />
            <button className="icon-button primary" onClick={() => runSearch(mode)} disabled={busy} title="Run search">
              <Play size={16} />
            </button>
          </div>
        </section>

        <section className="control-section">
          <div className="mode-grid">
            {modes.map((item) => {
              const Icon = item.icon;
              return (
                <button key={item.id} className={mode === item.id ? "mode active" : "mode"} onClick={() => runSearch(item.id)} disabled={busy}>
                  <Icon size={16} />
                  {item.label}
                </button>
              );
            })}
          </div>
        </section>

        <section className="control-section">
          <div className="section-title">
            <Radar size={16} />
            Filters
          </div>
          <div className="filter-grid">
            <NumberField label="MW max" value={filters.molecular_weight_max} onChange={(value) => setFilters({ ...filters, molecular_weight_max: value })} />
            <NumberField label="LogP max" value={filters.logp_max} onChange={(value) => setFilters({ ...filters, logp_max: value })} />
            <NumberField label="QED min" value={filters.qed_min} step={0.05} onChange={(value) => setFilters({ ...filters, qed_min: value })} />
            <NumberField label="Lipinski max" value={filters.lipinski_max} step={1} onChange={(value) => setFilters({ ...filters, lipinski_max: value })} />
          </div>
          <label htmlFor="targetClass">Target class</label>
          <select id="targetClass" value={filters.target_class ?? ""} onChange={(event) => setFilters({ ...filters, target_class: event.target.value || null })}>
            <option value="">Any</option>
            {targetClasses.map((klass) => (
              <option key={klass} value={klass}>
                {klass}
              </option>
            ))}
          </select>
          <label className="check-row">
            <input
              type="checkbox"
              checked={filters.exclude_high_toxicity}
              onChange={(event) => setFilters({ ...filters, exclude_high_toxicity: event.target.checked })}
            />
            Exclude high-risk heuristics
          </label>
          <label className="check-row">
            <input type="checkbox" checked={heatmap} onChange={(event) => setHeatmap(event.target.checked)} />
            Target-class heatmap
          </label>
        </section>

        <section className="control-section">
          <div className="section-title">
            <Sparkles size={16} />
            Discovery Steering
          </div>
          <Basket label="Toward" ids={positiveIds} onRemove={(id) => removeId(setPositiveIds, positiveIds, id)} />
          <Basket label="Away" ids={negativeIds} onRemove={(id) => removeId(setNegativeIds, negativeIds, id)} />
          <button className="full-button accent" onClick={runDiscovery} disabled={busy || positiveIds.length === 0}>
            <Sparkles size={16} />
            Steer Discovery
          </button>
        </section>

        <section className="control-section status-block">
          <div className="section-title">
            <Database size={16} />
            Qdrant Index
          </div>
          <p>{status ? `${status.count.toLocaleString()} points in ${status.mode} mode` : "Checking index"}</p>
          <button className="full-button" onClick={runBuild} disabled={busy}>
            <Database size={16} />
            Build / Refresh Index
          </button>
          <span className={busy ? "pulse status-pill" : "status-pill"}>{message}</span>
        </section>
      </aside>

      <section className="center-panel">
        <ChemicalMap
          points={mapPoints}
          results={response?.results ?? []}
          selectedId={selectedId}
          shortlistIds={shortlistIds}
          positiveIds={positiveIds}
          negativeIds={negativeIds}
          heatmap={heatmap}
          onSelect={setSelectedId}
        />
        <ResultStrip
          results={response?.results ?? []}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onShortlist={(id) => addUnique(setShortlistIds, shortlistIds, id)}
          onPositive={(id) => addUnique(setPositiveIds, positiveIds, id)}
          onNegative={(id) => addUnique(setNegativeIds, negativeIds, id)}
        />
      </section>

      <aside className="right-panel">
        <DetailPanel
          result={selectedResult}
          selectedId={selectedId}
          onShortlist={(id) => addUnique(setShortlistIds, shortlistIds, id)}
          onPositive={(id) => addUnique(setPositiveIds, positiveIds, id)}
          onNegative={(id) => addUnique(setNegativeIds, negativeIds, id)}
        />
        <QdrantPanel response={response} />
        <ShortlistPanel
          ids={shortlistIds}
          results={response?.results ?? []}
          compareLeft={compareLeft}
          compareRight={compareRight}
          compare={compare}
          onLeft={setCompareLeft}
          onRight={setCompareRight}
          onCompare={() => runCompare()}
          onRemove={(id) => removeId(setShortlistIds, shortlistIds, id)}
          onExport={downloadShortlist}
        />
      </aside>
    </main>
  );
}

function NumberField({ label, value, step = 1, onChange }: { label: string; value?: number | null; step?: number; onChange: (value: number | null) => void }) {
  return (
    <label>
      {label}
      <input
        type="number"
        step={step}
        value={value ?? ""}
        onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))}
      />
    </label>
  );
}

function Basket({ label, ids, onRemove }: { label: string; ids: string[]; onRemove: (id: string) => void }) {
  return (
    <div className="basket">
      <span>{label}</span>
      <div>
        {ids.length === 0 && <em>empty</em>}
        {ids.map((id) => (
          <button key={id} onClick={() => onRemove(id)}>
            {id}
          </button>
        ))}
      </div>
    </div>
  );
}

function ResultStrip({
  results,
  selectedId,
  onSelect,
  onShortlist,
  onPositive,
  onNegative,
}: {
  results: SearchResult[];
  selectedId?: string;
  onSelect: (id: string) => void;
  onShortlist: (id: string) => void;
  onPositive: (id: string) => void;
  onNegative: (id: string) => void;
}) {
  return (
    <section className="result-strip">
      {results.map((result, index) => (
        <article key={result.molecule.molecule_id} className={selectedId === result.molecule.molecule_id ? "result-card active" : "result-card"}>
          <button className="result-main" onClick={() => onSelect(result.molecule.molecule_id)}>
            <span>{index + 1}</span>
            <strong>{result.molecule.name}</strong>
            <small>{result.molecule.target_class}</small>
          </button>
          <div className="score-row">
            <b>{result.final_score.toFixed(2)}</b>
            <span>S {result.structure_similarity.toFixed(2)}</span>
            <span>B {result.bioactivity_similarity.toFixed(2)}</span>
          </div>
          {result.conflict_badge && <p className="badge warn">{result.conflict_badge}</p>}
          <div className="card-actions">
            <button title="Shortlist" onClick={() => onShortlist(result.molecule.molecule_id)}>
              <BookmarkPlus size={14} />
            </button>
            <button title="Positive example" onClick={() => onPositive(result.molecule.molecule_id)}>
              +
            </button>
            <button title="Negative example" onClick={() => onNegative(result.molecule.molecule_id)}>
              -
            </button>
          </div>
        </article>
      ))}
    </section>
  );
}

function DetailPanel({
  result,
  selectedId,
  onShortlist,
  onPositive,
  onNegative,
}: {
  result?: SearchResult;
  selectedId?: string;
  onShortlist: (id: string) => void;
  onPositive: (id: string) => void;
  onNegative: (id: string) => void;
}) {
  const molecule = result?.molecule;
  if (!molecule) {
    return (
      <section className="panel-card">
        <h2>Molecule Detail</h2>
        <p className="muted">Run a search to inspect molecular neighborhoods.</p>
      </section>
    );
  }
  return (
    <section className="panel-card">
      <div className="panel-heading">
        <div>
          <h2>{molecule.name}</h2>
          <p>{molecule.molecule_id}</p>
        </div>
        <span className={`tox ${molecule.toxicity_flag}`}>{molecule.toxicity_flag}</span>
      </div>
      <img className="structure-image" src={structureSvgUrl(molecule.molecule_id)} alt={`${molecule.name} molecular structure`} />
      <div className="metric-grid">
        <Metric label="QED" value={molecule.qed.toFixed(2)} />
        <Metric label="MW" value={molecule.molecular_weight.toFixed(1)} />
        <Metric label="LogP" value={molecule.logp.toFixed(2)} />
        <Metric label="TPSA" value={molecule.tpsa.toFixed(1)} />
        <Metric label="HBD/HBA" value={`${molecule.hbd}/${molecule.hba}`} />
        <Metric label="Lipinski" value={String(molecule.lipinski_violations)} />
      </div>
      <div className="detail-copy">
        <strong>{molecule.target_class}</strong>
        <span>{molecule.target_name}</span>
        <code>{molecule.canonical_smiles}</code>
      </div>
      <div className="why-list">
        <strong>Why this matched</strong>
        {result.why.map((reason) => (
          <span key={reason}>{reason}</span>
        ))}
      </div>
      <div className="button-row">
        <button onClick={() => onShortlist(selectedId ?? molecule.molecule_id)}>
          <BookmarkPlus size={15} />
          Shortlist
        </button>
        <button onClick={() => onPositive(selectedId ?? molecule.molecule_id)}>Toward</button>
        <button onClick={() => onNegative(selectedId ?? molecule.molecule_id)}>Away</button>
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function QdrantPanel({ response }: { response: SearchResponse | null }) {
  const qdrant = response?.qdrant;
  return (
    <section className="panel-card qdrant-panel">
      <div className="section-title">
        <Database size={16} />
        Qdrant Engine
      </div>
      {qdrant ? (
        <dl>
          <dt>Collection</dt>
          <dd>{String(qdrant.collection)}</dd>
          <dt>Query</dt>
          <dd>{String(qdrant.query)}</dd>
          <dt>Vectors</dt>
          <dd>{(qdrant.named_vectors as string[]).join(", ")} + {(qdrant.sparse_vectors as string[]).join(", ")}</dd>
          <dt>Candidates</dt>
          <dd>
            {String(qdrant.candidates_retrieved)} retrieved, {String(qdrant.final_reranked_results)} reranked
          </dd>
          <dt>Formula</dt>
          <dd>{String(qdrant.rerank_formula)}</dd>
        </dl>
      ) : (
        <p className="muted">Search results will show Qdrant query details here.</p>
      )}
    </section>
  );
}

function ShortlistPanel({
  ids,
  results,
  compareLeft,
  compareRight,
  compare,
  onLeft,
  onRight,
  onCompare,
  onRemove,
  onExport,
}: {
  ids: string[];
  results: SearchResult[];
  compareLeft: string;
  compareRight: string;
  compare: CompareResponse | null;
  onLeft: (id: string) => void;
  onRight: (id: string) => void;
  onCompare: () => void;
  onRemove: (id: string) => void;
  onExport: (format: "json" | "csv") => void;
}) {
  const options = Array.from(new Set(["CHEMBL25", "CHEMBL521", ...ids, ...results.map((result) => result.molecule.molecule_id)]));
  return (
    <section className="panel-card">
      <div className="section-title">
        <BookmarkPlus size={16} />
        Shortlist
      </div>
      <div className="shortlist">
        {ids.length === 0 && <p className="muted">No saved candidates yet.</p>}
        {ids.map((id) => (
          <button key={id} onClick={() => onRemove(id)}>
            {id}
          </button>
        ))}
      </div>
      <div className="button-row">
        <button onClick={() => onExport("json")} disabled={!ids.length}>
          <Download size={15} />
          JSON
        </button>
        <button onClick={() => onExport("csv")} disabled={!ids.length}>
          <Download size={15} />
          CSV
        </button>
      </div>
      <div className="compare-box">
        <div className="section-title">
          <GitCompare size={16} />
          Compare
        </div>
        <select value={compareLeft} onChange={(event) => onLeft(event.target.value)}>
          {options.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
        <select value={compareRight} onChange={(event) => onRight(event.target.value)}>
          {options.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
        <button className="full-button" onClick={onCompare}>
          <GitCompare size={16} />
          Compare Molecules
        </button>
        {compare && (
          <div className="compare-result">
            <span>Structure {compare.structure_similarity.toFixed(2)}</span>
            <span>Bioactivity {compare.bioactivity_similarity.toFixed(2)}</span>
            <span>Map distance {compare.map_distance.toFixed(1)}</span>
            <span>{compare.shared_target_class ? "shared target class" : "different target class"}</span>
          </div>
        )}
      </div>
    </section>
  );
}

export default App;
