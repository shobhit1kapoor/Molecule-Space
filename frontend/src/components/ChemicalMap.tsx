import { useMemo } from "react";
import * as d3 from "d3";
import type { MapPoint, SearchResult } from "../types/molspace";

interface ChemicalMapProps {
  points: MapPoint[];
  results: SearchResult[];
  selectedId?: string;
  shortlistIds: string[];
  positiveIds: string[];
  negativeIds: string[];
  heatmap: boolean;
  onSelect: (id: string) => void;
}

const targetColors: Record<string, string> = {
  enzyme: "#1b9aaa",
  gpcr: "#e4572e",
  "ion channel": "#7b61ff",
  transporter: "#2e7d32",
  "nuclear receptor": "#b23a8f",
  other: "#5f6c7b",
  unknown: "#5f6c7b",
};

export function ChemicalMap({
  points,
  results,
  selectedId,
  shortlistIds,
  positiveIds,
  negativeIds,
  heatmap,
  onSelect,
}: ChemicalMapProps) {
  const width = 920;
  const height = 620;
  const padding = 36;
  const resultById = useMemo(() => new Map(results.map((result, index) => [result.molecule.molecule_id, { result, index }])), [results]);
  const shortlist = useMemo(() => new Set(shortlistIds), [shortlistIds]);
  const positives = useMemo(() => new Set(positiveIds), [positiveIds]);
  const negatives = useMemo(() => new Set(negativeIds), [negativeIds]);

  const scales = useMemo(() => {
    const xDomain = d3.extent(points, (point) => point.x) as [number, number];
    const yDomain = d3.extent(points, (point) => point.y) as [number, number];
    return {
      x: d3.scaleLinear().domain(xDomain[0] === xDomain[1] ? [-1, 1] : xDomain).range([padding, width - padding]),
      y: d3.scaleLinear().domain(yDomain[0] === yDomain[1] ? [-1, 1] : yDomain).range([height - padding, padding]),
    };
  }, [points]);

  const classes = useMemo(() => Array.from(new Set(points.map((point) => point.target_class))).sort(), [points]);

  return (
    <section className="map-shell" aria-label="Chemical-space map">
      <div className="map-toolbar">
        <div>
          <strong>Chemical Space</strong>
          <span>{points.length.toLocaleString()} molecules</span>
        </div>
        <div className="legend">
          {classes.slice(0, 6).map((klass) => (
            <span key={klass}>
              <i style={{ background: targetColors[klass] ?? targetColors.other }} />
              {klass}
            </span>
          ))}
        </div>
      </div>
      <svg className="chemical-map" viewBox={`0 0 ${width} ${height}`} role="img">
        <rect x="0" y="0" width={width} height={height} rx="8" />
        {heatmap &&
          points.map((point) => (
            <circle
              key={`heat-${point.molecule_id}`}
              cx={scales.x(point.x)}
              cy={scales.y(point.y)}
              r={18 + point.qed * 10}
              fill={targetColors[point.target_class] ?? targetColors.other}
              opacity={0.055}
            />
          ))}
        {points.map((point) => {
          const matched = resultById.get(point.molecule_id);
          const isSelected = selectedId === point.molecule_id;
          const isShortlisted = shortlist.has(point.molecule_id);
          const isPositive = positives.has(point.molecule_id);
          const isNegative = negatives.has(point.molecule_id);
          const radius = Math.max(3.5, 4 + point.qed * 5 + (matched ? 2 : 0));
          const stroke =
            isPositive ? "#2e7d32" : isNegative ? "#b00020" : isSelected ? "#111827" : isShortlisted ? "#f59e0b" : "#ffffff";
          return (
            <g key={point.molecule_id} className="map-point-group">
              {matched && <circle cx={scales.x(point.x)} cy={scales.y(point.y)} r={radius + 10} className="result-glow" />}
              <circle
                className="map-point"
                cx={scales.x(point.x)}
                cy={scales.y(point.y)}
                r={radius}
                fill={targetColors[point.target_class] ?? targetColors.other}
                stroke={stroke}
                strokeWidth={isSelected || isPositive || isNegative ? 3 : 1.4}
                opacity={point.toxicity_flag === "high" ? 0.42 : 0.86}
                onClick={() => onSelect(point.molecule_id)}
              >
                <title>{`${point.name} | ${point.target_class} | QED ${point.qed}`}</title>
              </circle>
              {matched && matched.index < 8 && (
                <text x={scales.x(point.x) + 8} y={scales.y(point.y) - 8}>
                  {matched.index + 1}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </section>
  );
}

