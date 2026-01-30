"use client";

import React, { useEffect, useState } from "react";

type TextGenPipeline = (
  input: string,
  options?: {
    max_new_tokens?: number;
    temperature?: number;
    top_p?: number;
    do_sample?: boolean;
  }
) => Promise<Array<{ generated_text: string }>>;

function getErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

export default function Page() {
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);

  const [prompt, setPrompt] = useState("Once upon a time,");
  const [output, setOutput] = useState("");

  const [generator, setGenerator] = useState<TextGenPipeline | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadModel() {
      setLoading(true);
      setOutput("");

      try {
        const { pipeline } = await import("@huggingface/transformers");

        const gen = (await pipeline(
          "text-generation",
          "Xenova/llama2.c-stories15M"
        )) as unknown as TextGenPipeline;

        if (!cancelled) {
          setGenerator(() => gen);
          setReady(true);
        }
      } catch (err: unknown) {
        if (!cancelled) {
          setOutput(`Failed to load model.\n\n${getErrorMessage(err)}`);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadModel();
    return () => {
      cancelled = true;
    };
  }, []);

  async function generate() {
    if (!generator) return;

    setGenerating(true);
    setOutput("");

    try {
      const result = await generator(prompt, {
        max_new_tokens: 50,
        temperature: 0.8,
        top_p: 0.9,
        do_sample: true,
      });

      setOutput(result[0]?.generated_text ?? "");
    } catch (err: unknown) {
      setOutput(`Generation failed.\n\n${getErrorMessage(err)}`);
    } finally {
      setGenerating(false);
    }
  }

  return (
    <main style={{ padding: "2rem", maxWidth: 800, margin: "0 auto" }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 12 }}>
        llama2.c-stories15M (in-browser)
      </h1>

      <p style={{ marginBottom: 16 }}>
        Model: <code>Xenova/llama2.c-stories15M</code>
        <br />
        Status: {loading ? "loading…" : ready ? "ready" : "not ready"}
      </p>

      <label style={{ display: "block", fontWeight: 600, marginBottom: 8 }}>
        Prompt
      </label>

      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={5}
        style={{
          width: "100%",
          padding: 12,
          borderRadius: 8,
          border: "1px solid #ccc",
          marginBottom: 12,
          fontFamily:
            "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
        }}
      />

      <button
        onClick={generate}
        disabled={!ready || generating}
        style={{
          padding: "10px 14px",
          borderRadius: 8,
          border: "1px solid #ccc",
          cursor: !ready || generating ? "not-allowed" : "pointer",
          marginBottom: 16,
        }}
      >
        {generating ? "Generating…" : "Generate"}
      </button>

      <label style={{ display: "block", fontWeight: 600, marginBottom: 8 }}>
        Output
      </label>

      <pre
        style={{
          width: "100%",
          minHeight: 140,
          padding: 12,
          borderRadius: 8,
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
