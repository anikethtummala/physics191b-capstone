export type Basis = "x" | "z";
export type DecoderName = "mwpm" | "cnn" | "gnn" | "transformer";
export type ResultStatus = "complete" | "checkpoint_required" | "dependency_missing" | "error";
export type JobStatus = "pending" | "running" | "complete" | "error";

export const BENCHMARK_SUITE_ID = "surface-code-memory-v1";
export const BENCHMARK_SUITE_VERSION = "2026.04";

export interface NoiseSettings {
  p: number;
  after_clifford_depolarization?: number | null;
  before_round_data_depolarization?: number | null;
  before_measure_flip_probability?: number | null;
  after_reset_flip_probability?: number | null;
}

export interface BenchmarkRequest {
  suite_id: string;
  suite_version: string;
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
  suite_id?: string;
  suite_version?: string;
  suite_compliant?: boolean;
  distance: number;
  rounds: number;
  basis: Basis;
  noise_p: number;
  shots: number;
  case_id?: string | null;
  sample_seed?: number | null;
  trace_id?: string | null;
  circuit_sha256?: string | null;
  detection_events_sha256?: string | null;
  observable_flips_sha256?: string | null;
  status: ResultStatus;
  logical_error_rate: number | null;
  logical_error_rate_ci_low?: number | null;
  logical_error_rate_ci_high?: number | null;
  confidence_level?: number | null;
  confidence_method?: string | null;
  logical_errors: number | null;
  runtime_ms: number | null;
  runtime_us_per_shot: number | null;
  peak_memory_mb: number | null;
  model_parameters: number | null;
  error: string | null;
}

export interface BenchmarkJob {
  job_id: string;
  suite_id?: string;
  suite_version?: string;
  suite_compliant?: boolean;
  suite_compliance_errors?: string[];
  runtime_environment?: RuntimeEnvironment | null;
  status: JobStatus;
  progress: number;
  results: BenchmarkResult[];
  errors: string[];
}

export interface RuntimeEnvironment {
  api_version: string;
  python: string;
  platform: string;
  platform_release: string;
  machine: string;
  processor: string;
  cpu_count: number | null;
  dependencies: Record<string, string | null>;
}

export interface EncodedTraceArray {
  dtype: string;
  shape: number[];
  data_b64: string;
}

export interface TraceArtifact {
  case_id: string;
  sample_seed: number | null;
  circuit_sha256: string;
  detection_events_sha256: string;
  observable_flips_sha256: string;
  circuit_text: string;
  detection_events: EncodedTraceArray;
  observable_flips: EncodedTraceArray;
}

export interface BenchmarkSubmissionBundle {
  schema_version: string;
  suite_id: string;
  suite_version: string;
  suite_compliant: boolean;
  suite_compliance_errors: string[];
  job_id: string;
  request: BenchmarkRequest;
  runtime_environment: RuntimeEnvironment;
  results: BenchmarkResult[];
  traces: TraceArtifact[];
}

export interface SubmissionValidationResponse {
  valid: boolean;
  leaderboard_eligible: boolean;
  warnings: string[];
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
