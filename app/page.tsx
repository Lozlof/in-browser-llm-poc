"use client";

import { useEffect, useRef, useState } from "react";

type GeneratorFn = (prompt: string) => Promise<string>;

type TextGenItem = { generated_text: string };

function isTextGenItem(x: unknown): x is TextGenItem {
  return (
    typeof x === "object" &&
    x !== null &&
    "generated_text" in x &&
    typeof (x as Record<string, unknown>).generated_text === "string"
  );
}

export default function Page() {
  const generatorRef = useRef<GeneratorFn | null>(null);

  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [status, setStatus] = useState("Loading model...");
  const [prompt, setPrompt] = useState("Once upon a time,");
  const [output, setOutput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function init(): Promise<void> {
      try {
        setIsLoadingModel(true);
        setStatus("Importing transformers...");

        // Dynamic import ensures this runs ONLY in the browser
        const { pipeline } = await import("@huggingface/transformers");

        if (cancelled) return;
        setStatus(
          "Downloading/initializing model (first run may take a bit)..."
        );

        const pipe = await pipeline("text-generation", "Xenova/distilgpt2");

        if (cancelled) return;

        generatorRef.current = async (text: string): Promise<string> => {
          const result: unknown = await pipe(text, {
            max_new_tokens: 64,
            do_sample: true,
            temperature: 0.8,
            top_p: 0.9,
          });

          // Expected shape is usually: [{ generated_text: "..." }]
          if (Array.isArray(result) && result.length > 0 && isTextGenItem(result[0])) {
            return result[0].generated_text;
          }

          // Some versions can return a single object
          if (isTextGenItem(result)) {
            return result.generated_text;
          }

          // If something unexpected happens, return a debug-friendly string
          return typeof result === "string" ? result : JSON.stringify(result);
        };

        setStatus("Model ready.");
      } catch (e: unknown) {
        console.error(e);
        const message =
          e instanceof Error
            ? e.message
            : typeof e === "string"
            ? e
            : JSON.stringify(e);
        setStatus(`Error loading model: ${message}`);
      } finally {
        setIsLoadingModel(false);
      }
    }

    void init();

    return () => {
      cancelled = true;
    };
  }, []);

  async function onGenerate(): Promise<void> {
    const gen = generatorRef.current;
    if (!gen) return;

    setIsGenerating(true);
    setOutput("");

    try {
      const text = await gen(prompt);
      setOutput(text);
    } catch (e: unknown) {
      console.error(e);
      const message =
        e instanceof Error
          ? e.message
          : typeof e === "string"
          ? e
          : JSON.stringify(e);
      setOutput(`Error: ${message}`);
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <main
      style={{
        maxWidth: 900,
        margin: "40px auto",
        padding: 16,
        fontFamily: "system-ui",
      }}
    >
      <h1 style={{ fontSize: 28, marginBottom: 8 }}>
        DistilGPT-2 Browser POC
      </h1>
      <p style={{ marginTop: 0, opacity: 0.8 }}>
        Model: <code>Xenova/distilgpt2</code>
      </p>

      <div
        style={{
          padding: 12,
          border: "1px solid #ddd",
          borderRadius: 10,
          marginBottom: 16,
        }}
      >
        <strong>Status:</strong> {status}
      </div>

      <label style={{ display: "block", marginBottom: 8 }}>
        Input
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={8}
          style={{
            width: "100%",
            marginTop: 6,
            padding: 10,
            borderRadius: 10,
            border: "1px solid #ddd",
          }}
          placeholder="Type a prompt..."
          disabled={isLoadingModel || isGenerating}
        />
      </label>

      <button
        onClick={onGenerate}
        disabled={isLoadingModel || isGenerating}
        style={{
          padding: "10px 14px",
          borderRadius: 10,
          border: "1px solid #ddd",
          cursor: isLoadingModel || isGenerating ? "not-allowed" : "pointer",
          marginBottom: 16,
        }}
      >
        {isGenerating ? "Generating..." : "Generate"}
      </button>

      <label style={{ display: "block" }}>
        Output
        <textarea
          value={output}
          readOnly
          rows={10}
          style={{
            width: "100%",
            marginTop: 6,
            padding: 10,
            borderRadius: 10,
            border: "1px solid #ddd",
          }}
          placeholder="Model output will appear here..."
        />
      </label>
    </main>
  );
}
