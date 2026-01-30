"use client";

import { useEffect, useRef, useState } from "react";

type GeneratorFn = (prompt: string) => Promise<string>;

/**
 * The pipeline output types vary by transformers.js version and task input
 * (string vs chat messages). We'll treat as unknown and safely extract text.
 */
type ChatMessage = { role: string; content: string };

function isChatMessage(x: unknown): x is ChatMessage {
  return (
    typeof x === "object" &&
    x !== null &&
    "role" in x &&
    "content" in x &&
    typeof (x as Record<string, unknown>).role === "string" &&
    typeof (x as Record<string, unknown>).content === "string"
  );
}

function extractText(result: unknown): string {
  // Common shapes:
  // 1) [{ generated_text: "..." }]
  // 2) [{ generated_text: [ {role, content}, ... ] }]   (chat-style)
  // 3) { generated_text: ... }                          (less common)

  const getGeneratedText = (obj: unknown): unknown => {
    if (typeof obj === "object" && obj !== null && "generated_text" in obj) {
      return (obj as Record<string, unknown>).generated_text;
    }
    return undefined;
  };

  // If array, look at first item
  if (Array.isArray(result) && result.length > 0) {
    const gt = getGeneratedText(result[0]);
    if (typeof gt === "string") return gt;

    // Chat-style array of messages
    if (Array.isArray(gt) && gt.length > 0) {
      const last = gt[gt.length - 1];
      if (isChatMessage(last)) return last.content;
      // fallback stringify
      return JSON.stringify(gt);
    }
  }

  // If single object
  const gt = getGeneratedText(result);
  if (typeof gt === "string") return gt;
  if (Array.isArray(gt) && gt.length > 0) {
    const last = gt[gt.length - 1];
    if (isChatMessage(last)) return last.content;
    return JSON.stringify(gt);
  }

  // Fallback
  return typeof result === "string" ? result : JSON.stringify(result);
}

export default function Page() {
  const generatorRef = useRef<GeneratorFn | null>(null);

  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [status, setStatus] = useState("Loading model...");
  const [prompt, setPrompt] = useState(
    "Solve the equation: x^2 - 3x + 2 = 0"
  );
  const [output, setOutput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function init(): Promise<void> {
      try {
        setIsLoadingModel(true);
        setStatus("Importing Transformers.js...");

        // Dynamic import ensures client-only execution
        const { pipeline } = await import("@huggingface/transformers");

        if (cancelled) return;

        setStatus(
          "Downloading/initializing Phi-3 (first load can be large; WebGPU recommended)..."
        );

        const pipe = await pipeline(
          "text-generation",
          "Xenova/Phi-3-mini-4k-instruct"
        );

        if (cancelled) return;

        generatorRef.current = async (text: string): Promise<string> => {
          // Phi instruct is chat-tuned, so pass chat messages
          const messages = [{ role: "user", content: text }];

          const result: unknown = await pipe(messages, {
            max_new_tokens: 256,
            do_sample: false, // deterministic (faster & consistent)
          });

          return extractText(result);
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
        Phi-3 Mini (Browser) POC
      </h1>
      <p style={{ marginTop: 0, opacity: 0.8 }}>
        Model: <code>Xenova/Phi-3-mini-4k-instruct</code>
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
        <div style={{ marginTop: 8, opacity: 0.8, fontSize: 14 }}>
          Tip: For best results, use Chrome/Edge with WebGPU enabled.
        </div>
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