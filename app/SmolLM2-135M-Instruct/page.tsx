"use client";

import React, { useRef, useState } from "react";

type TextGenOptions = {
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  do_sample?: boolean;
};

type TextGenResult = Array<{ generated_text: unknown }>;

// Pipeline type with optional dispose()
type DisposablePipeline = ((
  input: string,
  options?: TextGenOptions
) => Promise<TextGenResult>) & {
  dispose?: () => Promise<void>;
};

function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

function extractText(result: TextGenResult): string {
  const v = result?.[0]?.generated_text;
  if (typeof v === "string") return v;

  try {
    return JSON.stringify(result, null, 2);
  } catch {
    return String(result);
  }
}

export default function Page() {
  const [status, setStatus] = useState<
    "idle" | "loading" | "running" | "error"
  >("idle");
  const [prompt, setPrompt] = useState(
    "Answer in one sentence: What is the capital of France?"
  );
  const [output, setOutput] = useState("");

  const pipelineRef = useRef<DisposablePipeline | null>(null);

  const modelId = "HuggingFaceTB/SmolLM2-135M-Instruct";

  async function loadPipeline(): Promise<DisposablePipeline> {
    if (pipelineRef.current) return pipelineRef.current;

    const { pipeline, env } = await import("@huggingface/transformers");

    // 🔒 HARD REQUIREMENTS
    // CPU-only
    const device = "wasm";

    // q4 quantization (will only work if model_q4.onnx exists)
    const dtype = "q4";

    // Optional safety knobs
    env.allowLocalModels = false;
    env.useBrowserCache = true; // cache ONNX files, not runtime tensors

    const p = (await pipeline("text-generation", modelId, {
      device,
      dtype,
    })) as unknown as DisposablePipeline;

    pipelineRef.current = p;
    return p;
  }

  async function disposePipeline() {
    const p = pipelineRef.current;
    pipelineRef.current = null;

    if (p?.dispose) {
      try {
        await p.dispose();
      } catch {
        // ignore dispose errors
      }
    }
  }

  async function runOnceAndFree() {
    setStatus("loading");
    setOutput("");

    try {
      const gen = await loadPipeline();
      setStatus("running");

      const result = await gen(prompt, {
        max_new_tokens: 64,
        temperature: 0.2,
        top_p: 0.9,
        do_sample: true,
      });

      setOutput(extractText(result));
      setStatus("idle");
    } catch (err: unknown) {
      setStatus("error");
      setOutput(`Generation failed.\n\n${getErrorMessage(err)}`);
    } finally {
      // 🔥 Blow everything away after the response
      await disposePipeline();
    }
  }

  return (
    <main style={{ padding: "2rem", maxWidth: 900, margin: "0 auto" }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 12 }}>
        SmolLM2-135M (CPU-only · q4 · disposable)
      </h1>

      <p style={{ marginBottom: 16 }}>
        Model: <code>{modelId}</code>
        <br />
        Backend: <code>wasm</code>
        <br />
        Quantization: <code>q4</code>
        <br />
        Status: <b>{status}</b>
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

      <button
        onClick={runOnceAndFree}
        disabled={status === "loading" || status === "running"}
        style={{
          padding: "10px 14px",
          borderRadius: 10,
          border: "1px solid #ccc",
          cursor:
            status === "loading" || status === "running"
              ? "not-allowed"
              : "pointer",
          marginBottom: 16,
        }}
      >
        {status === "running" ? "Generating…" : "Generate (CPU q4)"}
      </button>

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
