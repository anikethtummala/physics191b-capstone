import type {
  Basis,
  BenchmarkJob,
  BenchmarkRequest,
  HealthResponse,
  LayoutResponse,
  SampleResponse
} from "./types/api";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export function fetchLayout(distance: number, basis: Basis): Promise<LayoutResponse> {
  const params = new URLSearchParams({ distance: String(distance), basis });
  return request<LayoutResponse>(`/api/layout?${params.toString()}`);
}

export async function startBenchmark(payload: BenchmarkRequest): Promise<string> {
  const response = await request<{ job_id: string }>("/api/benchmarks", {
    method: "POST",
    body: JSON.stringify(payload)
  });
  return response.job_id;
}

export function fetchBenchmark(jobId: string): Promise<BenchmarkJob> {
  return request<BenchmarkJob>(`/api/benchmarks/${jobId}`);
}

export function fetchSample(payload: {
  distance: number;
  basis: Basis;
  noise: { p: number };
  seed: number;
}): Promise<SampleResponse> {
  return request<SampleResponse>("/api/sample", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}
