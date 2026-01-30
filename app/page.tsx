import Link from "next/link";

export default function Page() {
  return (
    <main style={{ padding: "2rem" }}>
      <ul>
        <li>
          <Link href="/DistilGPT-2">DistilGPT-2</Link>
        </li>
        <li>
          <Link href="/Phi-3-mini-4k-instruct">Phi-3-mini-4k-instruct</Link>
        </li>
        <li>
          <Link href="/SmolLM2-135M-Instruct">SmolLM2-135M-Instruct</Link>
        </li>
      </ul>
    </main>
  );
}