"use client";

import React, { useEffect, useMemo, useState } from "react";

type TextGenPipeline = (
  input: string,
  options?: {
    max_new_tokens?: number;
    temperature?: number;
    top_p?: number;
    do_sample?: boolean;
    repetition_penalty?: number;
  }
) => Promise<Array<{ generated_text: unknown }>>;

function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

function extractText(result: Array<{ generated_text: unknown }>): string {
  const v = result?.[0]?.generated_text;

  // Common case for plain text-generation: generated_text is a string
  if (typeof v === "string") return v;

  // Some models return arrays/objects (fallback)
  try {
    return JSON.stringify(result, null, 2);
  } catch {
    return String(result);
  }
}

export default function Page() {
  const [ready, setReady] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [generating, setGenerating] = useState(false);

  const [prompt, setPrompt] = useState(
    "Answer in one sentence: What is the capital of France?"
  );
  const [output, setOutput] = useState("");

  const [generator, setGenerator] = useState<TextGenPipeline | null>(null);

  const modelId = useMemo(() => "HuggingFaceTB/SmolLM2-135M-Instruct", []);

  async function loadModel() {
    setLoadingModel(true);
    setOutput("");

    try {
      const { pipeline, env } = await import("@huggingface/transformers");

      // Prefer WebGPU when available (faster, often better UX).
      // WASM can be smaller in some cases, but usually slower.
      // You can force one or the other.
      const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

      // Transformers.js settings knobs:
      env.allowLocalModels = false; // keep it simple
      // Optional: reduce caching behavior. If you want least disk usage:
      // env.useBrowserCache = false;

      const device = hasWebGPU ? "webgpu" : "wasm";

      const gen = (await pipeline("text-generation", modelId, {
        device,
      })) as unknown as TextGenPipeline;

      setGenerator(() => gen);
      setReady(true);
    } catch (err: unknown) {
      setOutput(`Failed to load model.\n\n${getErrorMessage(err)}`);
      setReady(false);
      setGenerator(null);
    } finally {
      setLoadingModel(false);
    }
  }

  useEffect(() => {
    // Load once up front. If you truly want “lowest footprint”,
    // you can remove this and only load on demand.
    void loadModel();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runOnceAndFree() {
    // Load on-demand if we previously freed it
    if (!generator) {
      await loadModel();
    }
    if (!generator) return;

    setGenerating(true);
    setOutput("");

    try {
      const result = await generator(prompt, {
        max_new_tokens: 64,
        temperature: 0.2,
        top_p: 0.9,
        do_sample: true,
        repetition_penalty: 1.05,
      });

      setOutput(extractText(result));
    } catch (err: unknown) {
      setOutput(`Generation failed.\n\n${getErrorMessage(err)}`);
    } finally {
      setGenerating(false);

      // Drop references so GC can reclaim *everything it can*.
      // This won’t always instantly shrink memory (GC is nondeterministic),
      // but it is the correct “free it” signal in JS.
      setGenerator(null);
      setReady(false);
    }
  }

  return (
    <main style={{ padding: "2rem", maxWidth: 900, margin: "0 auto" }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 12 }}>
        SmolLM2 (min-footprint mode)
      </h1>

      <p style={{ marginBottom: 16 }}>
        Model: <code>{modelId}</code>
        <br />
        Status:{" "}
        {loadingModel ? "loading model…" : ready ? "ready" : "not loaded"}
      </p>

      <label style={{ display: "block", fontWeight: 600, marginBottom: 8 }}>
        Prompt
      </label>

      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={6}
        style={{
          width: "100%",
          padding: 12,
          borderRadius: 10,
          border: "1px solid #ccc",
          marginBottom: 12,
          fontFamily:
            "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
        }}
      />

      <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
        <button
          onClick={runOnceAndFree}
          disabled={generating || loadingModel}
          style={{
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid #ccc",
            cursor: generating || loadingModel ? "not-allowed" : "pointer",
          }}
        >
          {generating ? "Generating…" : "Generate (then free model)"}
        </button>

        <button
          onClick={() => {
            // Manual “panic free”
            setGenerator(null);
            setReady(false);
            setOutput("");
          }}
          style={{
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid #ccc",
            cursor: "pointer",
          }}
        >
          Clear
        </button>
      </div>

      <label style={{ display: "block", fontWeight: 600, marginBottom: 8 }}>
        Output
      </label>

      <pre
        style={{
          width: "100%",
          minHeight: 160,
          padding: 12,
          borderRadius: 10,
          border: "1px solid #ccc",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {output || "—"}
      </pre>
    </main>
  );
}
