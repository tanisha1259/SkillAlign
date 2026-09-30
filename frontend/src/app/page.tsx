"use client";

import { ChangeEvent, FormEvent, useState } from "react";

type Result = {
  requirement: string;
  requirement_type: string;
  importance: string;
  status: "Strong" | "Partial" | "Unsupported";
  evidence: string | null;
  gap_type: string | null;
  gap: string | null;
  priority: string;
  confidence: number;
};

type AnalysisResponse = {
  requirements_analyzed: number;
  match_score: number;
  summary: {
    strong: number;
    partial: number;
    unsupported: number;
    high_priority_gaps: number;
  };
  results: Result[];
};

export default function Home() {
  const [resume, setResume] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleResumeChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;

    if (file && file.type !== "application/pdf") {
      setError("Please upload a PDF resume.");
      setResume(null);
      return;
    }

    setError("");
    setResume(file);
  };

  const analyzeResume = async (event: FormEvent) => {
    event.preventDefault();

    if (!resume) {
      setError("Please upload your resume.");
      return;
    }

    if (!jobDescription.trim()) {
      setError("Please enter the job description.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("resume", resume);
      formData.append("job_description", jobDescription);

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/analyze`,{
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Analysis failed.");
      }

      const data: AnalysisResponse = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the resume.",
      );
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setResult(null);
    setError("");
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/95">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <button
            onClick={resetAnalysis}
            className="text-xl font-bold tracking-tight"
          >
            Skill<span className="text-indigo-400">Align</span>
          </button>

          <div className="hidden text-sm text-slate-400 sm:block">
            Explainable Resume–JD Matching
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-6 py-12">
        {!result ? (
          <>
            {/* Hero */}
            <section className="mx-auto max-w-3xl text-center">
              <div className="mb-4 inline-flex rounded-full border border-indigo-500/30 bg-indigo-500/10 px-4 py-2 text-sm text-indigo-300">
                AI-powered resume analysis
              </div>

              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
                Understand how your resume matches a job.
              </h1>

              <p className="mt-5 text-lg leading-8 text-slate-400">
                SkillAlign compares your resume against a job description,
                evaluates each requirement, and identifies the skills and
                experience you may be missing.
              </p>
            </section>

            {/* Input form */}
            <form
              onSubmit={analyzeResume}
              className="mx-auto mt-12 max-w-6xl"
            >
              <div className="grid gap-6 lg:grid-cols-2">
                {/* Resume upload */}
                <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
                  <div className="mb-5">
                    <h2 className="text-lg font-semibold">Your Resume</h2>
                    <p className="mt-1 text-sm text-slate-400">
                      Upload your resume as a PDF.
                    </p>
                  </div>

                  <label
                    htmlFor="resume"
                    className="flex min-h-64 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 bg-slate-950/60 px-6 text-center transition hover:border-indigo-500 hover:bg-indigo-500/5"
                  >
                    <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-indigo-500/10 text-2xl">
                      📄
                    </div>

                    {resume ? (
                      <>
                        <p className="font-medium text-white">{resume.name}</p>
                        <p className="mt-2 text-sm text-slate-400">
                          {(resume.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                        <p className="mt-4 text-sm text-indigo-400">
                          Click to change file
                        </p>
                      </>
                    ) : (
                      <>
                        <p className="font-medium text-white">
                          Upload PDF resume
                        </p>
                        <p className="mt-2 text-sm text-slate-500">
                          Click to browse your files
                        </p>
                      </>
                    )}

                    <input
                      id="resume"
                      type="file"
                      accept=".pdf,application/pdf"
                      onChange={handleResumeChange}
                      className="hidden"
                    />
                  </label>
                </section>

                {/* JD input */}
                <section className="rounded-2xl border border-slate-800 bg-slate-900 p-6">
                  <div className="mb-5">
                    <h2 className="text-lg font-semibold">
                      Job Description
                    </h2>
                    <p className="mt-1 text-sm text-slate-400">
                      Paste the job description you want to analyze.
                    </p>
                  </div>

                  <textarea
                    value={jobDescription}
                    onChange={(event) =>
                      setJobDescription(event.target.value)
                    }
                    placeholder="Paste the job description here..."
                    className="min-h-64 w-full resize-none rounded-xl border border-slate-700 bg-slate-950 p-4 text-sm leading-6 text-white outline-none transition placeholder:text-slate-600 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  />
                </section>
              </div>

              {/* Error */}
              {error && (
                <div className="mt-6 rounded-xl border border-red-500/30 bg-red-500/10 px-5 py-4 text-sm text-red-300">
                  {error}
                </div>
              )}

              {/* Analyze button */}
              <div className="mt-8 flex justify-center">
                <button
                  type="submit"
                  disabled={loading}
                  className="rounded-xl bg-indigo-500 px-10 py-4 font-semibold text-white shadow-lg shadow-indigo-500/20 transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loading ? "Analyzing resume..." : "Analyze Resume"}
                </button>
              </div>
            </form>
          </>
        ) : (
          <AnalysisResults result={result} onReset={resetAnalysis} />
        )}
      </div>
    </main>
  );
}

function AnalysisResults({
  result,
  onReset,
}: {
  result: AnalysisResponse;
  onReset: () => void;
}) {
  const score = result.match_score;

  return (
    <section>
      {/* Results header */}
      <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
        <div>
          <div className="mb-3 inline-flex rounded-full border border-indigo-500/30 bg-indigo-500/10 px-4 py-2 text-sm text-indigo-300">
            Analysis complete
          </div>

          <h1 className="text-3xl font-bold sm:text-4xl">
            Resume–JD Match Analysis
          </h1>

          <p className="mt-2 text-slate-400">
            {result.requirements_analyzed} requirements analyzed.
          </p>
        </div>

        <button
          onClick={onReset}
          className="rounded-lg border border-slate-700 px-5 py-3 text-sm font-medium text-slate-300 transition hover:border-slate-500 hover:text-white"
        >
          Analyze another job
        </button>
      </div>

      {/* Score + breakdown */}
      <div className="mt-8 grid gap-6 lg:grid-cols-3">
        {/* Score */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-8 lg:col-span-1">
          <p className="text-sm font-medium uppercase tracking-wider text-slate-500">
            Resume–JD Match
          </p>

          <div className="mt-4 flex items-end gap-2">
            <span className="text-6xl font-bold tracking-tight text-white">
              {score.toFixed(2)}
            </span>
            <span className="mb-2 text-2xl text-slate-500">%</span>
          </div>

          <div className="mt-6 h-3 overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-indigo-500 transition-all"
              style={{ width: `${Math.min(Math.max(score, 0), 100)}%` }}
            />
          </div>
        </div>

        {/* Breakdown */}
        <div className="grid grid-cols-3 gap-3 lg:col-span-2">
          <SummaryCard
            label="Strong"
            value={result.summary.strong}
            description="Clearly supported"
            className="border-emerald-500/20 bg-emerald-500/5"
            valueClass="text-emerald-400"
          />

          <SummaryCard
            label="Partial"
            value={result.summary.partial}
            description="Partially supported"
            className="border-amber-500/20 bg-amber-500/5"
            valueClass="text-amber-400"
          />

          <SummaryCard
            label="Unsupported"
            value={result.summary.unsupported}
            description="No sufficient evidence"
            className="border-red-500/20 bg-red-500/5"
            valueClass="text-red-400"
          />
        </div>
      </div>

      {/* High priority gaps */}
      {result.summary.high_priority_gaps > 0 && (
        <div className="mt-6 rounded-2xl border border-red-500/20 bg-red-500/5 p-6">
          <div className="flex items-center gap-3">
            <span className="flex h-9 w-9 items-center justify-center rounded-full bg-red-500/10">
              !
            </span>

            <div>
              <h2 className="font-semibold text-white">
                {result.summary.high_priority_gaps} high-priority skill gaps
              </h2>
              <p className="text-sm text-slate-400">
                These requirements currently have no sufficient supporting
                evidence in the resume.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Requirements */}
      <div className="mt-8">
        <div className="mb-5">
          <h2 className="text-2xl font-bold">Requirement Analysis</h2>
          <p className="mt-1 text-sm text-slate-400">
            Requirement-level evidence and skill-gap analysis.
          </p>
        </div>

        <div className="space-y-4">
          {result.results.map((item, index) => (
            <RequirementCard key={`${item.requirement}-${index}`} item={item} />
          ))}
        </div>
      </div>
    </section>
  );
}

function SummaryCard({
  label,
  value,
  description,
  className,
  valueClass,
}: {
  label: string;
  value: number;
  description: string;
  className: string;
  valueClass: string;
}) {
  return (
    <div className={`rounded-2xl border p-6 ${className}`}>
      <p className="text-sm font-medium text-slate-400">{label}</p>
      <p className={`mt-3 text-4xl font-bold ${valueClass}`}>{value}</p>
      <p className="mt-2 text-xs text-slate-500">{description}</p>
    </div>
  );
}

function RequirementCard({ item }: { item: Result }) {
  const isStrong = item.status === "Strong";
  const isPartial = item.status === "Partial";

  const statusClass = isStrong
    ? "border-emerald-500/20 bg-emerald-500/5 text-emerald-400"
    : isPartial
      ? "border-amber-500/20 bg-amber-500/5 text-amber-400"
      : "border-red-500/20 bg-red-500/5 text-red-400";

  const icon = isStrong ? "✓" : isPartial ? "~" : "✕";

  return (
    <article className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900">
      {/* Requirement header */}
      <div className="p-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex min-w-0 gap-4">
            <div
              className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full border font-bold ${statusClass}`}
            >
              {icon}
            </div>

            <div className="min-w-0">
              <h3 className="font-medium leading-6 text-white">
                {item.requirement}
              </h3>

              <div className="mt-3 flex flex-wrap gap-2">
                <span className="rounded-md bg-slate-800 px-2.5 py-1 text-xs capitalize text-slate-400">
                  {item.requirement_type.replaceAll("_", " ")}
                </span>

                <span className="rounded-md bg-slate-800 px-2.5 py-1 text-xs capitalize text-slate-400">
                  {item.importance}
                </span>

                {item.priority !== "none" && (
                  <span className="rounded-md bg-red-500/10 px-2.5 py-1 text-xs capitalize text-red-400">
                    {item.priority} priority
                  </span>
                )}
              </div>
            </div>
          </div>

          <span
            className={`w-fit shrink-0 rounded-full border px-3 py-1.5 text-xs font-semibold ${statusClass}`}
          >
            {item.status}
          </span>
        </div>
      </div>

      {/* Evidence */}
      {item.evidence && (
        <div className="border-t border-slate-800 bg-slate-950/40 px-6 py-5">
          <div className="mb-2 flex items-center gap-2">
            <span className="text-sm">↳</span>
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Resume Evidence
            </p>
          </div>

          <p className="text-sm leading-6 text-slate-300">
            {item.evidence}
          </p>
        </div>
      )}

      {/* Gap */}
      {item.gap && (
        <div className="border-t border-red-500/10 bg-red-500/5 px-6 py-5">
          <div className="mb-2 flex items-center gap-2">
            <span className="text-sm text-red-400">!</span>
            <p className="text-xs font-semibold uppercase tracking-wider text-red-400">
              Skill Gap
            </p>
          </div>

          <p className="text-sm leading-6 text-slate-300">{item.gap}</p>
        </div>
      )}
    </article>
  );
}