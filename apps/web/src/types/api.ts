export type Basis = "x" | "z";
export type DecoderName = "mwpm" | "cnn" | "gnn" | "transformer";
export type ResultStatus = "complete" | "checkpoint_required" | "dependency_missing" | "error";
export type JobStatus = "pending" | "running" | "complete" | "error";

export interface NoiseSettings {
  p: number;
  after_clifford_depolarization?: number | null;
  before_round_data_depolarization?: number | null;
  before_measure_flip_probability?: number | null;
  after_reset_flip_probability?: number | null;
}

export interface BenchmarkRequest {
  distances: number[];
  rounds?: number | null;
  basis: Basis;
  noise: NoiseSettings;
  shots: number;
  seed?: number | null;
  decoders: DecoderName[];
}

export interface BenchmarkResult {
  decoder: DecoderName;
  distance: number;
  rounds: number;
  basis: Basis;
  noise_p: number;
  shots: number;
  status: ResultStatus;
  logical_error_rate: number | null;
  logical_errors: number | null;
  runtime_ms: number | null;
  runtime_us_per_shot: number | null;
  peak_memory_mb: number | null;
  model_parameters: number | null;
  error: string | null;
}

export interface BenchmarkJob {
  job_id: string;
  status: JobStatus;
  progress: number;
  results: BenchmarkResult[];
  errors: string[];
}

export interface Qubit {
  id: string;
  index: number;
  kind: "data" | "measure";
  x: number;
  y: number;
  stabilizer_type: Basis | null;
  boundary: boolean;
}

export interface LayoutEdge {
  source: string;
  target: string;
  kind: "stabilizer" | "logical";
}

export interface LayoutResponse {
  distance: number;
  basis: Basis;
  surface17: boolean;
  data_qubits: number;
  measure_qubits: number;
  bounds: Record<"min_x" | "min_y" | "max_x" | "max_y", number>;
  qubits: Qubit[];
  edges: LayoutEdge[];
}

export interface SyndromeEvent {
  detector: number;
  x: number;
  y: number;
  t: number;
  stabilizer_type: Basis | null;
}

export interface MatchEdge {
  source: number;
  target: number | null;
  boundary: boolean;
}

export interface SampleResponse {
  distance: number;
  rounds: number;
  basis: Basis;
  noise_p: number;
  logical_observable_flip: boolean;
  mwpm_prediction: boolean | null;
  using_fallback: boolean;
  events: SyndromeEvent[];
  matches: MatchEdge[];
  error: string | null;
}

export interface HealthResponse {
  ok: boolean;
  quantum_stack: boolean;
  missing: string[];
}
