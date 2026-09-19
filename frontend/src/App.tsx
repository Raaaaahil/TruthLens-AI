import { useEffect, useMemo, useState, type ChangeEvent, type DragEvent, type ReactNode } from "react";

type Token = {
  token: string;
  importance: number;
};

type Analysis = {
  prediction: string;
  confidence: number;
  fake_probability: number;
  real_probability: number;
  important_tokens?: Token[];
};

type OCRSegment = {
  text: string;
  confidence: number;
  x: number;
  y: number;
};

type VerificationClaim = {
  claim: string;
  assessment?: string;
  support_score?: number;
  refute_score?: number;
  evidence?: Array<{
    title?: string;
    url?: string;
    text?: string;
    domain?: string;
    nli_label?: string;
    nli_score?: number;
  }>;
};

type Verification = {
  claims_checked?: number;
  claims?: VerificationClaim[];
  overall_assessment?: string;
  overall_score?: number;
  note?: string;
  status?: string;
  verification_id?: string;
  [key: string]: unknown;
};

type ImageResult = {
  source: string;
  filename: string;
  ocr: {
    text: string;
    segments: OCRSegment[];
    segment_count: number;
    ocr_language_hint?: string;
    ocr_language_name?: string;
    ocr_language_confidence?: number;
  };
  model_analysis?: Analysis;
  analysis?: Analysis;
  explanation?: Analysis;
  verification?: Verification;
  verification_id?: string;
  language?: { code: string; name: string; confidence: number };
  analysis_language?: { code: string; name: string };
  translation_applied?: boolean;
  original_text?: string;
  translated_text?: string;
  pipeline: string[];
};

type MultilingualResult = {
  source: "multilingual";
  language: { code: string; name: string; confidence: number };
  analysis_language: { code: string; name: string };
  translation_applied: boolean;
  original_text: string;
  translated_text: string;
  analysis: Analysis;
  verification?: Verification;
  verification_id?: string;
  pipeline: string[];
};

type URLResult = {
  source: string;
  url: string;
  article: {
    title: string;
    text: string;
    word_count: number;
    character_count: number;
  };
  model_analysis?: Analysis;
  analysis?: Analysis;
  explanation?: Analysis;
  verification?: Verification;
  verification_id?: string;
  pipeline: string[];
};

const API = "http://127.0.0.1:8000";

type HistoryItem = {
  id: string;
  timestamp: string;
  source: "text" | "image" | "url" | "multilingual";
  title: string;
  snippet: string;
  prediction: string;
  confidence: number;
  fake_probability: number;
  real_probability: number;
  important_tokens: Token[];
  verification?: Verification | null;
  articleText?: string;
  url?: string;
};

function getTextFromResult(
  sourceText: string,
  urlResult: URLResult | null,
  ocrResult: ImageResult["ocr"] | null
) {
  if (urlResult) return urlResult.article.text || "";
  if (ocrResult) return ocrResult.text || "";
  return sourceText;
}

function heuristicChecks(text: string) {
  const t = text.trim();
  const words = t.split(/\s+/).filter(Boolean);
  const upperWords = words.filter(
    (w) => w.length >= 4 && w === w.toUpperCase()
  ).length;
  const exclamations = (t.match(/!/g) || []).length;
  const questions = (t.match(/\?/g) || []).length;

  const sensational = [
    "shocking",
    "breaking",
    "urgent",
    "secret",
    "exposed",
    "miracle",
    "unbelievable",
    "you won't believe",
    "must see",
    "viral",
    "destroyed",
    "bombshell",
    "scandal",
    "terrifying",
    "alert",
    "huge",
  ];

  const emotional = [
    "outrage",
    "angry",
    "fear",
    "terrified",
    "disaster",
    "horrific",
    "amazing",
    "evil",
    "traitor",
    "insane",
    "crazy",
  ];

  const lower = t.toLowerCase();
  const sensationalHits = sensational.filter((x) => lower.includes(x));
  const emotionalHits = emotional.filter((x) => lower.includes(x));

  const clickbaitScore = Math.min(
    100,
    sensationalHits.length * 14 +
      Math.min(exclamations, 4) * 10 +
      (questions >= 2 ? 12 : 0)
  );

  const styleScore = Math.min(
    100,
    emotionalHits.length * 12 +
      Math.min(upperWords, 5) * 10 +
      Math.min(exclamations, 4) * 8
  );

  const aiSignals = [
    /in conclusion/i.test(t),
    /it is important to note/i.test(t),
    /overall,?\s/i.test(t),
    /furthermore/i.test(t),
    /moreover/i.test(t),
    words.length > 180,
    !/[.!?]/.test(t),
  ].filter(Boolean).length;

  const aiSignalScore = Math.min(100, aiSignals * 12);

  return {
    clickbaitScore,
    styleScore,
    aiSignalScore,
    sensationalHits,
    emotionalHits,
    uppercaseRatio: words.length ? upperWords / words.length : 0,
    exclamations,
    wordCount: words.length,
  };
}

function scoreLabel(score: number) {
  if (score >= 70) return "High";
  if (score >= 40) return "Moderate";
  return "Low";
}

function safePercent(value?: number) {
  if (typeof value !== "number" || Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(100, value * 100));
}

function Card({
  title,
  children,
  className = "",
}: {
  title?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-3xl border border-white/10 bg-white/[0.03] p-6 shadow-2xl backdrop-blur-xl ${className}`}
    >
      {title && <h3 className="mb-5 text-xl font-bold">{title}</h3>}
      {children}
    </div>
  );
}

function Metric({
  label,
  value,
  sub,
  tone = "cyan",
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "cyan" | "red" | "green" | "amber";
}) {
  const colors = {
    cyan: "text-cyan-400",
    red: "text-red-400",
    green: "text-emerald-400",
    amber: "text-amber-400",
  };

  return (
    <Card>
      <p className="text-sm text-slate-500">{label}</p>
      <p className={`mt-3 text-3xl font-bold ${colors[tone]}`}>{value}</p>
      {sub && <p className="mt-2 text-xs text-slate-500">{sub}</p>}
    </Card>
  );
}

function DashboardView({
  history,
  onOpen,
  onClear,
  onNewAnalysis,
}: {
  history: HistoryItem[];
  onOpen: (item: HistoryItem) => void;
  onClear: () => void;
  onNewAnalysis: () => void;
}) {
  const total = history.length;
  const fake = history.filter((x) => x.prediction === "FAKE").length;
  const real = history.filter((x) => x.prediction === "REAL").length;

  const validConfidences = history
    .map((x) => Number(x.confidence))
    .filter((value) => Number.isFinite(value));

  const avg = validConfidences.length
    ? validConfidences.reduce((sum, value) => sum + value, 0) /
      validConfidences.length
    : 0;

  const textCount = history.filter((x) => x.source === "text").length;
  const imageCount = history.filter((x) => x.source === "image").length;
  const urlCount = history.filter((x) => x.source === "url").length;
  const multilingualCount = history.filter(
    (x) => x.source === "multilingual"
  ).length;

  const fakePct = total ? (fake / total) * 100 : 0;
  const realPct = total ? (real / total) * 100 : 0;

  return (
    <div>
      <div className="mb-10 flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="mb-3 text-sm font-medium uppercase tracking-[0.3em] text-cyan-400">
            TruthLens Analytics
          </p>
          <h2 className="text-4xl font-bold tracking-tight md:text-5xl">
            Analysis Dashboard
          </h2>
          <p className="mt-3 max-w-2xl text-slate-400">
            Review recent news analyses, predictions and confidence statistics.
          </p>
        </div>

        <button
          onClick={onNewAnalysis}
          className="rounded-xl bg-cyan-500 px-5 py-3 font-semibold text-slate-950 hover:bg-cyan-400"
        >
          + New Analysis
        </button>
      </div>

      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          label="Total Analyses"
          value={String(total)}
          sub="Saved locally in this browser"
        />
        <Metric
          label="Fake Predictions"
          value={String(fake)}
          tone="red"
          sub={`${fakePct.toFixed(1)}% of analyses`}
        />
        <Metric
          label="Real Predictions"
          value={String(real)}
          tone="green"
          sub={`${realPct.toFixed(1)}% of analyses`}
        />
        <Metric
          label="Average Confidence"
          value={`${(avg * 100).toFixed(1)}%`}
          tone="cyan"
          sub="Across saved analyses"
        />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Card title="Prediction Distribution">
          <div className="space-y-5">
            <div>
              <div className="mb-2 flex justify-between text-sm">
                <span className="text-red-400">FAKE</span>
                <span>
                  {fake} · {fakePct.toFixed(1)}%
                </span>
              </div>
              <div className="h-3 rounded-full bg-white/10">
                <div
                  className="h-3 rounded-full bg-red-400"
                  style={{ width: `${fakePct}%` }}
                />
              </div>
            </div>

            <div>
              <div className="mb-2 flex justify-between text-sm">
                <span className="text-emerald-400">REAL</span>
                <span>
                  {real} · {realPct.toFixed(1)}%
                </span>
              </div>
              <div className="h-3 rounded-full bg-white/10">
                <div
                  className="h-3 rounded-full bg-emerald-400"
                  style={{ width: `${realPct}%` }}
                />
              </div>
            </div>
          </div>
        </Card>

        <Card title="Input Sources">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              ["📝", textCount, "Text"],
              ["📷", imageCount, "Images"],
              ["🌐", urlCount, "URLs"],
              ["🌍", multilingualCount, "Multilingual"],
            ].map(([icon, count, label]) => (
              <div
                key={String(label)}
                className="rounded-2xl border border-white/10 bg-slate-950/50 p-4 text-center"
              >
                <p className="text-2xl">{icon}</p>
                <p className="mt-2 text-2xl font-bold text-cyan-400">
                  {String(count)}
                </p>
                <p className="text-xs text-slate-500">{String(label)}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card className="mt-6">
        <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-xl font-bold">Recent Analyses</h3>
            <p className="mt-1 text-sm text-slate-500">
              Latest 50 analyses stored locally.
            </p>
          </div>

          {total > 0 && (
            <button
              onClick={onClear}
              className="rounded-xl border border-red-400/20 px-4 py-2 text-sm text-red-400 hover:bg-red-400/10"
            >
              Clear History
            </button>
          )}
        </div>

        {total === 0 ? (
          <div className="rounded-2xl border border-dashed border-white/10 bg-slate-950/40 p-12 text-center">
            <div className="text-5xl">📊</div>
            <p className="mt-4 text-lg font-semibold">No analyses yet</p>
            <p className="mt-2 text-sm text-slate-500">
              Run an analysis and it will appear here.
            </p>
            <button
              onClick={onNewAnalysis}
              className="mt-5 rounded-xl bg-cyan-500 px-5 py-3 text-sm font-semibold text-slate-950"
            >
              Start Analyzing
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {history.map((item) => (
              <button
                key={item.id}
                onClick={() => onOpen(item)}
                className="w-full rounded-2xl border border-white/10 bg-slate-950/40 p-4 text-left transition hover:border-cyan-400/30 hover:bg-cyan-400/[0.03]"
              >
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                      <span className="rounded-full bg-cyan-400/10 px-2.5 py-1 text-cyan-400">
                        {item.source === "url"
                          ? "🌐 URL"
                          : item.source === "image"
                            ? "📷 IMAGE"
                            : item.source === "multilingual"
                              ? "🌍 MULTILINGUAL"
                              : "📝 TEXT"}
                      </span>

                      <span
                        className={`rounded-full px-2.5 py-1 ${
                          item.prediction === "FAKE"
                            ? "bg-red-400/10 text-red-400"
                            : "bg-emerald-400/10 text-emerald-400"
                        }`}
                      >
                        {item.prediction}
                      </span>
                    </div>

                    <p className="truncate font-semibold text-slate-200">
                      {item.title || "Untitled analysis"}
                    </p>

                    <p className="mt-1 line-clamp-2 text-xs leading-5 text-slate-500">
                      {item.snippet}
                    </p>
                  </div>

                  <div className="shrink-0 text-left md:text-right">
                    <p className="text-lg font-bold text-cyan-400">
                      {(item.confidence * 100).toFixed(1)}%
                    </p>
                    <p className="mt-1 text-xs text-slate-600">
                      {new Date(item.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </Card>

      <div className="mt-6 rounded-2xl border border-amber-400/10 bg-amber-400/[0.03] px-5 py-4 text-xs leading-6 text-slate-500">
        <span className="font-medium text-amber-400">Privacy note:</span>{" "}
        History is stored only in this browser's local storage.
      </div>
    </div>
  );
}

function App() {
  const [mode, setMode] = useState<
    "text" | "image" | "url" | "multilingual"
  >("text");

  const [multilingualResult, setMultilingualResult] =
    useState<MultilingualResult | null>(null);

  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const [result, setResult] = useState<Analysis | null>(null);
  const [ocrResult, setOcrResult] =
    useState<ImageResult["ocr"] | null>(null);
  const [imageMultilingualResult, setImageMultilingualResult] =
    useState<ImageResult | null>(null);
  const [urlResult, setUrlResult] = useState<URLResult | null>(null);
  const [verification, setVerification] =
    useState<Verification | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [showArticle, setShowArticle] = useState(false);
  const [view, setView] = useState<"analyzer" | "dashboard">("analyzer");

  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const saved = localStorage.getItem("truthlens_history");
      if (!saved) return [];

      const parsed = JSON.parse(saved);
      if (!Array.isArray(parsed)) return [];

      return parsed.map((item) => ({
        ...item,
        prediction:
          item?.prediction === "FAKE" || item?.prediction === "REAL"
            ? item.prediction
            : Number(item?.fake_probability) >
                Number(item?.real_probability)
              ? "FAKE"
              : "REAL",

        confidence: Number.isFinite(Number(item?.confidence))
          ? Number(item.confidence)
          : Number.isFinite(Number(item?.real_probability))
            ? Number(item.real_probability)
            : Number.isFinite(Number(item?.fake_probability))
              ? Number(item.fake_probability)
              : 0,

        fake_probability: Number.isFinite(
          Number(item?.fake_probability)
        )
          ? Number(item.fake_probability)
          : 0,

        real_probability: Number.isFinite(
          Number(item?.real_probability)
        )
          ? Number(item.real_probability)
          : 0,

        important_tokens: Array.isArray(item?.important_tokens)
          ? item.important_tokens
          : [],
      }));
    } catch {
      return [];
    }
  });

  const sourceText = getTextFromResult(text, urlResult, ocrResult);

  const heuristics = useMemo(
    () => heuristicChecks(sourceText),
    [sourceText]
  );

  const saveHistory = (
    item: Omit<HistoryItem, "id" | "timestamp">
  ) => {
    const entry: HistoryItem = {
      ...item,
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp: new Date().toISOString(),
    };

    setHistory((current) => {
      const updated = [entry, ...current].slice(0, 50);

      try {
        localStorage.setItem(
          "truthlens_history",
          JSON.stringify(updated)
        );
      } catch {}

      return updated;
    });
  };

  const clearHistory = () => {
    setHistory([]);

    try {
      localStorage.removeItem("truthlens_history");
    } catch {}
  };

  const openHistoryItem = (item: HistoryItem) => {
    setView("analyzer");
    setMode(item.source);

    setText(
      item.source === "text" || item.source === "multilingual"
        ? item.articleText || item.snippet
        : ""
    );

    setUrl(item.url || "");

    setResult({
      prediction: item.prediction,
      confidence: item.confidence,
      fake_probability: item.fake_probability,
      real_probability: item.real_probability,
      important_tokens: item.important_tokens || [],
    });

    setVerification(item.verification || null);
    setMultilingualResult(null);
    setImageMultilingualResult(null);
    setOcrResult(null);

    setUrlResult(
      item.source === "url"
        ? {
            source: "url",
            url: item.url || "",
            article: {
              title: item.title,
              text: item.articleText || item.snippet,
              word_count: (item.articleText || "")
                .split(/\s+/)
                .filter(Boolean).length,
              character_count: (item.articleText || "").length,
            },
            model_analysis: {
              prediction: item.prediction,
              confidence: item.confidence,
              fake_probability: item.fake_probability,
              real_probability: item.real_probability,
              important_tokens: item.important_tokens || [],
            },
            explanation: {
              prediction: item.prediction,
              confidence: item.confidence,
              fake_probability: item.fake_probability,
              real_probability: item.real_probability,
              important_tokens: item.important_tokens || [],
            },
            verification: item.verification || undefined,
            pipeline: [
              "History",
              "BERT Classification",
              "XAI Explanation",
            ],
          }
        : null
    );

    setError("");
  };

  const changeMode = (
    newMode: "text" | "image" | "url" | "multilingual"
  ) => {
    setMode(newMode);
    setError("");
    setResult(null);
    setOcrResult(null);
    setImageMultilingualResult(null);
    setUrlResult(null);
    setMultilingualResult(null);
    setVerification(null);
    setShowArticle(false);
  };

  const handleImageSelect = (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setImage(file);
    setImagePreview(URL.createObjectURL(file));
    setResult(null);
    setOcrResult(null);
    setImageMultilingualResult(null);
    setUrlResult(null);
    setMultilingualResult(null);
    setVerification(null);
    setError("");
  };

  const handleFileChange = (
    event: ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file) handleImageSelect(file);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);

    const file = event.dataTransfer.files?.[0];
    if (file) handleImageSelect(file);
  };

  const analyzeText = async () => {
    if (!text.trim()) {
      setError("Please enter some news text.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setVerification(null);
    setOcrResult(null);
    setUrlResult(null);

    try {
      const [explainResponse, verifyResponse] = await Promise.all([
        fetch(`${API}/explain`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: text.trim(),
          }),
        }),

        fetch(`${API}/verify`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: text.trim(),
          }),
        }),
      ]);

      const explainData = await explainResponse.json();
      const verifyData = await verifyResponse.json();

      if (!explainResponse.ok || explainData.error) {
        throw new Error(
          explainData.detail ||
            explainData.error ||
            "Failed to analyze text."
        );
      }

      const analysis: Analysis =
        explainData.analysis || explainData;

      const verificationResult =
        verifyResponse.ok && !verifyData.error
          ? verifyData.verification || null
          : null;

      setResult(analysis);
      setVerification(verificationResult);

      saveHistory({
        source: "text",
        title: text
          .trim()
          .split(/\s+/)
          .slice(0, 12)
          .join(" "),
        snippet: text.trim().slice(0, 180),
        prediction: analysis.prediction,
        confidence: analysis.confidence,
        fake_probability: analysis.fake_probability,
        real_probability: analysis.real_probability,
        important_tokens: analysis.important_tokens || [],
        verification: verificationResult,
        articleText: text.trim(),
      });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  };

  const analyzeMultilingual = async () => {
    if (!text.trim()) {
      setError(
        "Please enter Hindi, Urdu, English or mixed-language news text."
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setVerification(null);
    setOcrResult(null);
    setImageMultilingualResult(null);
    setUrlResult(null);
    setMultilingualResult(null);

    try {
      const response = await fetch(
        `${API}/analyze-multilingual`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: text.trim(),
          }),
        }
      );

      const data: MultilingualResult & {
        error?: string;
        detail?: string;
      } = await response.json();

      if (!response.ok || data.error) {
        throw new Error(
          data.detail ||
            data.error ||
            "Failed to analyze multilingual text."
        );
      }

      if (!data.analysis) {
        throw new Error(
          "Multilingual analysis result was not returned by the server."
        );
      }

      setMultilingualResult(data);
      setResult(data.analysis);
      setVerification(data.verification || null);

      saveHistory({
        source: "multilingual",
        title: `${data.language.name} news analysis`,
        snippet: data.original_text.slice(0, 180),
        prediction: data.analysis.prediction,
        confidence: data.analysis.confidence,
        fake_probability: data.analysis.fake_probability,
        real_probability: data.analysis.real_probability,
        important_tokens:
          data.analysis.important_tokens || [],
        verification: data.verification || null,
        articleText: data.original_text,
      });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  };

  const analyzeImage = async () => {
    if (!image) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setOcrResult(null);
    setImageMultilingualResult(null);
    setUrlResult(null);
    setVerification(null);

    try {
      const formData = new FormData();
      formData.append("file", image);

      const response = await fetch(`${API}/analyze-image`, {
        method: "POST",
        body: formData,
      });

      const data: ImageResult & {
        error?: string;
        detail?: string;
      } = await response.json();

      if (!response.ok || data.error) {
        throw new Error(
          data.detail ||
            data.error ||
            "Failed to analyze image."
        );
      }

      setOcrResult(data.ocr);
      setImageMultilingualResult(data);

      const analysis =
        data.analysis ||
        data.model_analysis ||
        data.explanation;

      if (!analysis) {
        throw new Error(
          "Analysis result was not returned by the server."
        );
      }

      const verificationResult =
        data.verification || {
          status: "PENDING",
          overall_assessment: "PENDING",
          claims_checked: 0,
          claims: [],
          note: "Evidence verification is running in the background.",
        };

      setResult(analysis);
      setVerification(verificationResult);

      saveHistory({
        source: "image",
        title: data.filename || "News image",
        snippet: data.ocr.text.slice(0, 180),
        prediction: analysis.prediction,
        confidence: analysis.confidence,
        fake_probability: analysis.fake_probability,
        real_probability: analysis.real_probability,
        important_tokens:
          analysis.important_tokens || [],
        verification: verificationResult,
        articleText: data.ocr.text,
      });

      // Image analysis is already complete. Start claim verification
      // separately so OCR/BERT/XAI results stay visible immediately.
      // The current backend exposes /verify as the verification endpoint
      // for text, so the OCR-extracted/translated text is sent there.
      const verificationText =
        data.translated_text ||
        data.original_text ||
        data.ocr.text;

      void (async () => {
        try {
          const verifyResponse = await fetch(`${API}/verify`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              text: verificationText,
            }),
          });

          const verifyData = await verifyResponse.json();

          if (!verifyResponse.ok || verifyData.error) {
            return;
          }

          const completedVerification =
            verifyData.verification || null;

          if (!completedVerification) return;

          setVerification(completedVerification);

          setImageMultilingualResult((current) =>
            current
              ? {
                  ...current,
                  verification: completedVerification,
                }
              : current
          );

          setHistory((current) => {
            const updated = current.map((item) =>
              item.source === "image" &&
              item.title === (data.filename || "News image") &&
              item.articleText === data.ocr.text
                ? {
                    ...item,
                    verification: completedVerification,
                  }
                : item
            );

            try {
              localStorage.setItem(
                "truthlens_history",
                JSON.stringify(updated)
              );
            } catch {}

            return updated;
          });
        } catch {
          // Keep the completed image/BERT result visible even if
          // evidence verification is temporarily unavailable.
        }
      })();
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  };

  const analyzeURL = async () => {
    if (!url.trim()) {
      setError("Please enter an article URL.");
      return;
    }

    if (!/^https?:\/\//i.test(url.trim())) {
      setError(
        "Please enter a valid HTTP or HTTPS URL."
      );
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setOcrResult(null);
    setUrlResult(null);
    setVerification(null);

    try {
      const response = await fetch(`${API}/analyze-url`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url: url.trim(),
        }),
      });

      const data: URLResult & {
        error?: string;
        detail?: string;
      } = await response.json();

      if (!response.ok || data.error) {
        throw new Error(
          data.detail ||
            data.error ||
            "Failed to analyze URL."
        );
      }

      setUrlResult(data);

      // Authoritative backend analysis first.
      // XAI explanation is only a fallback.
      const analysis =
        data.analysis ||
        data.model_analysis ||
        data.explanation;

      if (!analysis) {
        throw new Error(
          "Analysis result was not returned by the server."
        );
      }

      const verificationResult =
        data.verification || null;

      setResult(analysis);
      setVerification(verificationResult);

      saveHistory({
        source: "url",
        title: data.article.title || data.url,
        snippet: data.article.text.slice(0, 180),
        prediction: analysis.prediction,
        confidence: analysis.confidence,
        fake_probability: analysis.fake_probability,
        real_probability: analysis.real_probability,
        important_tokens:
          analysis.important_tokens || [],
        verification: verificationResult,
        articleText: data.article.text,
        url: data.url,
      });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const isUrlMode = mode === "url";
    const isImageMode = mode === "image";

    if (!verification || (!isUrlMode && !isImageMode)) {
      return;
    }

    const status = String(
      verification.status || ""
    ).toUpperCase();

    if (
      !["PENDING", "PROCESSING", "RUNNING"].includes(
        status
      )
    ) {
      return;
    }

    const verificationId =
      verification.verification_id ||
      urlResult?.verification_id ||
      imageMultilingualResult?.verification_id;

    const targetUrl =
      urlResult?.url ||
      (verification.url as string | undefined);

    // Image jobs are identified by verification_id only.
    // URL jobs keep the existing URL fallback for compatibility.
    if (!verificationId && (!isUrlMode || !targetUrl)) {
      return;
    }

    let stopped = false;
    let attempts = 0;
    let timer: number | null = null;

    const poll = async () => {
      if (stopped || attempts >= 300) {
        return;
      }

      attempts += 1;

      try {
        let response: Response | null = null;

        if (verificationId) {
          response = await fetch(
            `${API}/verification-status/${encodeURIComponent(
              verificationId
            )}`
          );
        }

        if (
          isUrlMode &&
          (!response || !response.ok) &&
          targetUrl
        ) {
          response = await fetch(
            `${API}/verification-status`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                url: targetUrl,
              }),
            }
          );
        }

        if (!response || !response.ok) {
          if (!stopped) {
            timer = window.setTimeout(poll, 2000);
          }
          return;
        }

        const data: Verification =
          await response.json();

        if (stopped) return;

        const nextStatus = String(
          data.status || ""
        ).toUpperCase();

        if (
          ["COMPLETED", "ERROR"].includes(nextStatus) ||
          (data.claims_checked || 0) > 0
        ) {
          setVerification(data);

          if (isUrlMode) {
            setUrlResult((current) =>
              current
                ? {
                    ...current,
                    verification: data,
                  }
                : current
            );
          }

          if (isImageMode) {
            setImageMultilingualResult((current) =>
              current
                ? {
                    ...current,
                    verification: data,
                    verification_id:
                      String(
                        data.verification_id ||
                        current.verification_id ||
                        ""
                      ),
                  }
                : current
            );
          }

          setHistory((current) => {
            const updated = current.map((item) => {
              const sameJob =
                item.verification?.verification_id ===
                verificationId;

              const sameUrl =
                isUrlMode &&
                targetUrl &&
                item.url === targetUrl;

              return sameJob || sameUrl
                ? {
                    ...item,
                    verification: data,
                  }
                : item;
            });

            try {
              localStorage.setItem(
                "truthlens_history",
                JSON.stringify(updated)
              );
            } catch {}

            return updated;
          });

          return;
        }
      } catch {}

      if (!stopped) {
        timer = window.setTimeout(poll, 2000);
      }
    };

    timer = window.setTimeout(poll, 1200);

    return () => {
      stopped = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [
    mode,
    urlResult?.url,
    urlResult?.verification_id,
    imageMultilingualResult?.verification_id,
    verification?.status,
    verification?.verification_id,
  ]);

  const clearAll = () => {
    setText("");
    setUrl("");
    setImage(null);
    setImagePreview(null);
    setResult(null);
    setOcrResult(null);
    setImageMultilingualResult(null);
    setUrlResult(null);
    setMultilingualResult(null);
    setVerification(null);
    setError("");
    setShowArticle(false);
  };

  const pipeline =
    multilingualResult?.pipeline ||
    imageMultilingualResult?.pipeline ||
    urlResult?.pipeline ||
    (ocrResult
      ? [
          "Image Upload",
          "EasyOCR",
          "Text Extraction",
          "BERT Classification",
          "XAI Explanation",
          "Claim Extraction",
          "Evidence Retrieval",
          "NLI Verification",
        ]
      : [
          "Text Input",
          "BERT Classification",
          "XAI Explanation",
          "Claim Extraction",
          "Evidence Retrieval",
          "NLI Verification",
        ]);

  const credibility = useMemo(() => {
    if (!result) return null;

    const modelConfidence = safePercent(
      result.confidence
    );

    const claims = verification?.claims || [];

    const checked = Number(
      verification?.claims_checked ||
        claims.length ||
        0
    );

    const verified = claims.filter((c) =>
      [
        "SUPPORTED",
        "ENTAILMENT",
        "VERIFIED",
      ].includes(
        (c.assessment || "").toUpperCase()
      )
    ).length;

    const evidenceCoverage = checked
      ? (verified / checked) * 100
      : 0;

    const sourceCount = new Set(
      claims
        .flatMap((c) =>
          (c.evidence || []).map(
            (e) => e.domain || e.url || ""
          )
        )
        .filter(Boolean)
    ).size;

    const diversity = Math.min(
      100,
      sourceCount * 25
    );

    const score = Math.round(
      modelConfidence * 0.55 +
        evidenceCoverage * 0.3 +
        diversity * 0.15
    );

    return {
      score: Math.max(
        0,
        Math.min(100, score)
      ),
      modelConfidence,
      evidenceCoverage,
      sourceCount,
    };
  }, [result, verification]);

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <header className="sticky top-0 z-40 border-b border-white/10 bg-slate-950/90 backdrop-blur-2xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-4 sm:px-6">
          <button
            onClick={() => setView("analyzer")}
            className="flex items-center gap-3 text-left"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/20 bg-cyan-400/10 text-lg font-black text-cyan-400 shadow-lg shadow-cyan-500/10">
              TL
            </div>

            <div>
              <h1 className="text-lg font-bold tracking-tight sm:text-xl">
                TruthLens{" "}
                <span className="text-cyan-400">
                  AI
                </span>
              </h1>

              <p className="hidden text-xs text-slate-500 sm:block">
                Explainable News Intelligence
              </p>
            </div>
          </button>

          <nav className="hidden items-center gap-1 rounded-xl border border-white/10 bg-white/[0.03] p-1 md:flex">
            <button
              onClick={() =>
                setView("analyzer")
              }
              className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                view === "analyzer"
                  ? "bg-cyan-500 text-slate-950"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Analyzer
            </button>

            <button
              onClick={() =>
                setView("dashboard")
              }
              className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
                view === "dashboard"
                  ? "bg-cyan-500 text-slate-950"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Dashboard
            </button>
          </nav>

          <div className="flex items-center gap-2">
            <span className="flex items-center gap-2 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-2 text-xs font-medium text-emerald-400">
              <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
              Online
            </span>

            <span className="hidden rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-2 text-xs text-cyan-400 sm:inline-flex">
              BERT · XAI · OCR
            </span>
          </div>
        </div>

        <div className="mx-auto flex max-w-7xl px-4 pb-3 md:hidden">
          <div className="grid w-full grid-cols-2 rounded-xl border border-white/10 bg-white/[0.03] p-1">
            <button
              onClick={() =>
                setView("analyzer")
              }
              className={`rounded-lg px-3 py-2 text-xs font-medium ${
                view === "analyzer"
                  ? "bg-cyan-500 text-slate-950"
                  : "text-slate-400"
              }`}
            >
              Analyzer
            </button>

            <button
              onClick={() =>
                setView("dashboard")
              }
              className={`rounded-lg px-3 py-2 text-xs font-medium ${
                view === "dashboard"
                  ? "bg-cyan-500 text-slate-950"
                  : "text-slate-400"
              }`}
            >
              Dashboard
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 sm:py-10">
        {view === "dashboard" ? (
          <DashboardView
            history={history}
            onOpen={openHistoryItem}
            onClear={clearHistory}
            onNewAnalysis={() =>
              setView("analyzer")
            }
          />
        ) : (
          <>
            <div className="relative mb-8 overflow-hidden rounded-3xl border border-cyan-400/10 bg-gradient-to-br from-cyan-400/[0.07] via-white/[0.025] to-transparent px-5 py-10 sm:px-8 sm:py-12">
              <div className="pointer-events-none absolute left-1/2 top-0 h-40 w-72 -translate-x-1/2 rounded-full bg-cyan-400/10 blur-3xl" />

              <div className="relative text-center">
                <div className="mx-auto mb-4 flex w-fit items-center gap-2 rounded-full border border-cyan-400/15 bg-cyan-400/5 px-4 py-2 text-xs font-medium uppercase tracking-[0.22em] text-cyan-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
                  AI-Powered News Analysis
                </div>

                <h2 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">
                  See beyond the headline.
                  <span className="mt-2 block text-cyan-400">
                    Understand the evidence.
                  </span>
                </h2>

                <p className="mx-auto mt-5 max-w-2xl text-sm leading-7 text-slate-400 sm:text-base">
                  Analyze article text, screenshots and public
                  URLs with BERT classification, explainable AI,
                  OCR and independent claim verification.
                </p>

                <div className="mt-6 flex flex-wrap justify-center gap-2 text-xs text-slate-500">
                  <span className="rounded-full border border-white/10 bg-black/20 px-3 py-1.5">
                    BERT V2
                  </span>
                  <span className="rounded-full border border-white/10 bg-black/20 px-3 py-1.5">
                    Explainable AI
                  </span>
                  <span className="rounded-full border border-white/10 bg-black/20 px-3 py-1.5">
                    OCR
                  </span>
                  <span className="rounded-full border border-white/10 bg-black/20 px-3 py-1.5">
                    Evidence Verification
                  </span>
                </div>
              </div>
            </div>

            <section className="rounded-3xl border border-white/10 bg-white/[0.035] p-4 shadow-2xl shadow-black/20 backdrop-blur-xl sm:p-6">
              <div className="mb-5 flex flex-col gap-2 border-b border-white/10 pb-5 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm font-semibold text-slate-200">
                    Choose an analysis method
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    Text, URL, image or multilingual input.
                  </p>
                </div>

                <span className="w-fit rounded-full border border-white/10 bg-slate-950/60 px-3 py-1.5 text-xs text-slate-500">
                  Analysis workspace
                </span>
              </div>

              <div className="mb-6 grid grid-cols-2 gap-1 rounded-2xl border border-white/10 bg-black/25 p-1.5 sm:grid-cols-4">
                {[
                  ["text", "📝 Text"],
                  ["url", "🌐 URL"],
                  ["image", "📷 Image"],
                  ["multilingual", "🌍 Multilingual"],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    onClick={() =>
                      changeMode(
                        value as
                          | "text"
                          | "image"
                          | "url"
                          | "multilingual"
                      )
                    }
                    className={`rounded-lg px-4 py-3 text-sm font-medium transition ${
                      mode === value
                        ? "bg-cyan-500 text-slate-950"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {mode === "multilingual" && (
                <div>
                  <div className="mb-5 rounded-2xl border border-cyan-400/10 bg-cyan-400/[0.03] p-5">
                    <div className="flex items-start gap-4">
                      <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-cyan-400/10 text-2xl">
                        🌍
                      </div>

                      <div>
                        <h3 className="font-semibold text-slate-200">
                          Multilingual News Analysis
                        </h3>

                        <p className="mt-1 text-sm leading-6 text-slate-500">
                          Automatically detects Hindi, Urdu or
                          English, translates Hindi/Urdu to
                          English, then runs the same BERT + XAI
                          pipeline.
                        </p>
                      </div>
                    </div>
                  </div>

                  <label className="mb-3 block text-sm font-medium text-slate-300">
                    Enter Multilingual News Text
                  </label>

                  <textarea
                    value={text}
                    onChange={(e) =>
                      setText(e.target.value)
                    }
                    placeholder={`उदाहरण: सरकार ने आज एक नई नीति की घोषणा की।

Example Urdu: حکومت نے آج ایک نئی پالیسی کا اعلان کیا۔`}
                    className="min-h-[240px] w-full resize-y rounded-2xl border border-white/10 bg-slate-900/80 p-5 text-slate-200 outline-none placeholder:text-slate-600 focus:border-cyan-400/50 focus:ring-2 focus:ring-cyan-400/10"
                  />

                  <div className="mt-4 flex gap-3">
                    <button
                      onClick={
                        analyzeMultilingual
                      }
                      disabled={loading}
                      className="flex-1 rounded-xl bg-cyan-500 px-6 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {loading
                        ? "Detecting, Translating & Analyzing..."
                        : "Analyze Multilingual News"}
                    </button>

                    <button
                      onClick={clearAll}
                      className="rounded-xl border border-white/10 px-6 py-3 font-medium text-slate-300 transition hover:bg-white/5"
                    >
                      Clear
                    </button>
                  </div>

                  {multilingualResult && (
                    <div className="mt-6 grid gap-4 md:grid-cols-3">
                      <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                        <p className="text-xs text-slate-500">
                          Detected Language
                        </p>

                        <p className="mt-2 text-xl font-semibold text-cyan-400">
                          {multilingualResult.language.name}
                        </p>

                        <p className="mt-1 text-xs text-slate-500">
                          Confidence{" "}
                          {(
                            multilingualResult.language
                              .confidence * 100
                          ).toFixed(1)}
                          %
                        </p>
                      </div>

                      <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                        <p className="text-xs text-slate-500">
                          Analysis Language
                        </p>

                        <p className="mt-2 text-xl font-semibold">
                          {
                            multilingualResult
                              .analysis_language.name
                          }
                        </p>
                      </div>

                      <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                        <p className="text-xs text-slate-500">
                          Translation
                        </p>

                        <p className="mt-2 text-xl font-semibold text-emerald-400">
                          {multilingualResult.translation_applied
                            ? "Applied"
                            : "Not Required"}
                        </p>
                      </div>
                    </div>
                  )}

                  {multilingualResult && (
                    <div className="mt-4 grid gap-4 md:grid-cols-2">
                      <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-5">
                        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-cyan-400">
                          Original Text
                        </p>

                        <p className="whitespace-pre-wrap leading-7 text-slate-300">
                          {
                            multilingualResult.original_text
                          }
                        </p>
                      </div>

                      <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-5">
                        <p className="mb-3 text-xs font-medium uppercase tracking-wider text-emerald-400">
                          English for BERT Analysis
                        </p>

                        <p className="whitespace-pre-wrap leading-7 text-slate-300">
                          {
                            multilingualResult.translated_text
                          }
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {mode === "text" && (
                <div>
                  <label className="mb-3 block text-sm font-medium text-slate-300">
                    Enter News Article
                  </label>

                  <textarea
                    value={text}
                    onChange={(e) =>
                      setText(e.target.value)
                    }
                    placeholder="Paste a news article or headline here..."
                    className="min-h-[240px] w-full resize-y rounded-2xl border border-white/10 bg-slate-900/80 p-5 text-slate-200 outline-none placeholder:text-slate-600 focus:border-cyan-400/50 focus:ring-2 focus:ring-cyan-400/10"
                  />

                  <div className="mt-4 flex gap-3">
                    <button
                      onClick={analyzeText}
                      disabled={loading}
                      className="flex-1 rounded-xl bg-cyan-500 px-6 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {loading
                        ? "Analyzing..."
                        : "Analyze News"}
                    </button>

                    <button
                      onClick={clearAll}
                      className="rounded-xl border border-white/10 px-6 py-3 font-medium text-slate-300 transition hover:bg-white/5"
                    >
                      Clear
                    </button>
                  </div>
                </div>
              )}

              {mode === "image" && (
                <div>
                  <label className="mb-3 block text-sm font-medium text-slate-300">
                    Upload News Screenshot
                  </label>

                  <div
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragActive(true);
                    }}
                    onDragLeave={() =>
                      setDragActive(false)
                    }
                    onDrop={handleDrop}
                    className={`relative rounded-2xl border-2 border-dashed p-8 text-center transition ${
                      dragActive
                        ? "border-cyan-400 bg-cyan-400/10"
                        : "border-white/10 bg-slate-900/50 hover:border-cyan-400/40"
                    }`}
                  >
                    {imagePreview ? (
                      <div className="space-y-5">
                        <img
                          src={imagePreview}
                          alt="Uploaded news"
                          className="mx-auto max-h-[400px] max-w-full rounded-xl object-contain"
                        />

                        <div>
                          <p className="font-medium text-slate-200">
                            {image?.name}
                          </p>

                          <p className="mt-1 text-sm text-slate-500">
                            {image
                              ? (
                                  image.size / 1024
                                ).toFixed(1)
                              : "0"}{" "}
                            KB
                          </p>
                        </div>

                        <label className="inline-block cursor-pointer rounded-lg border border-white/10 px-4 py-2 text-sm text-slate-300 hover:bg-white/5">
                          Change Image

                          <input
                            type="file"
                            accept="image/png,image/jpeg,image/webp"
                            onChange={
                              handleFileChange
                            }
                            className="hidden"
                          />
                        </label>
                      </div>
                    ) : (
                      <div className="py-10">
                        <div className="mb-4 text-5xl">
                          📷
                        </div>

                        <p className="text-lg font-medium">
                          Drag & Drop your news image
                        </p>

                        <p className="mt-2 text-sm text-slate-500">
                          or
                        </p>

                        <label className="mt-4 inline-block cursor-pointer rounded-xl bg-white/10 px-5 py-3 text-sm font-medium transition hover:bg-white/15">
                          Browse Image

                          <input
                            type="file"
                            accept="image/png,image/jpeg,image/webp"
                            onChange={
                              handleFileChange
                            }
                            className="hidden"
                          />
                        </label>

                        <p className="mt-4 text-xs text-slate-600">
                          JPG, PNG or WEBP
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="mt-4 flex gap-3">
                    <button
                      onClick={analyzeImage}
                      disabled={
                        loading || !image
                      }
                      className="flex-1 rounded-xl bg-cyan-500 px-6 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {loading
                        ? "Extracting & Analyzing..."
                        : "Analyze Image"}
                    </button>

                    <button
                      onClick={clearAll}
                      className="rounded-xl border border-white/10 px-6 py-3 font-medium text-slate-300 transition hover:bg-white/5"
                    >
                      Clear
                    </button>
                  </div>
                </div>
              )}

              {mode === "url" && (
                <div>
                  <label className="mb-3 block text-sm font-medium text-slate-300">
                    Enter News Article URL
                  </label>

                  <div className="rounded-2xl border border-white/10 bg-slate-900/80 p-5">
                    <div className="flex items-center gap-3">
                      <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-cyan-400/10 text-2xl">
                        🌐
                      </div>

                      <input
                        type="url"
                        value={url}
                        onChange={(e) =>
                          setUrl(e.target.value)
                        }
                        onKeyDown={(e) => {
                          if (
                            e.key === "Enter" &&
                            !loading
                          ) {
                            analyzeURL();
                          }
                        }}
                        placeholder="https://example.com/news/article"
                        className="w-full bg-transparent py-3 text-slate-200 outline-none placeholder:text-slate-600"
                      />
                    </div>
                  </div>

                  <p className="mt-3 text-xs text-slate-500">
                    Paste a public news article URL.
                    TruthLens will extract readable
                    article content before analysis.
                  </p>

                  <div className="mt-4 flex gap-3">
                    <button
                      onClick={analyzeURL}
                      disabled={
                        loading || !url.trim()
                      }
                      className="flex-1 rounded-xl bg-cyan-500 px-6 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {loading
                        ? "Extracting & Analyzing..."
                        : "Analyze URL"}
                    </button>

                    <button
                      onClick={clearAll}
                      className="rounded-xl border border-white/10 px-6 py-3 font-medium text-slate-300 transition hover:bg-white/5"
                    >
                      Clear
                    </button>
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-5 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-300">
                  ⚠️ {error}
                </div>
              )}
            </section>

            {urlResult && (
              <Card className="mt-8">
                <div className="mb-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-400">
                      🌐 URL SOURCE
                    </span>

                    <h3 className="mt-3 text-xl font-bold">
                      Extracted Article
                    </h3>

                    <p className="mt-1 text-sm text-slate-500">
                      {urlResult.article.word_count}{" "}
                      words ·{" "}
                      {
                        urlResult.article
                          .character_count
                      }{" "}
                      characters
                    </p>
                  </div>

                  <button
                    onClick={() =>
                      setShowArticle((v) => !v)
                    }
                    className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-300 hover:bg-white/5"
                  >
                    {showArticle
                      ? "Hide Article"
                      : "View Article"}
                  </button>
                </div>

                <div className="rounded-2xl border border-cyan-400/10 bg-cyan-400/[0.03] p-5">
                  <p className="mb-2 text-xs font-medium uppercase tracking-wider text-cyan-400">
                    Article Title
                  </p>

                  <h4 className="text-lg font-semibold leading-7 text-slate-200">
                    {urlResult.article.title ||
                      "Title not detected"}
                  </h4>
                </div>

                {showArticle && (
                  <div className="mt-5 rounded-2xl border border-white/10 bg-slate-950/60 p-5">
                    <p className="mb-3 text-sm font-medium text-slate-300">
                      Extracted Content
                    </p>

                    <p className="whitespace-pre-wrap leading-7 text-slate-300">
                      {urlResult.article.text}
                    </p>
                  </div>
                )}
              </Card>
            )}

            {ocrResult && (
              <Card className="mt-8">
                <div className="mb-5 flex items-center justify-between">
                  <div>
                    <h3 className="text-xl font-bold">
                      OCR Extracted Text
                    </h3>

                    <p className="mt-1 text-sm text-slate-500">
                      {ocrResult.segment_count} text
                      regions detected
                    </p>
                  </div>

                  <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-xs text-cyan-400">
                    EasyOCR
                  </span>
                </div>

                <div className="rounded-2xl border border-white/10 bg-slate-950/60 p-5">
                  <p className="whitespace-pre-wrap leading-7 text-slate-300">
                    {ocrResult.text}
                  </p>
                </div>
              </Card>
            )}

            {imageMultilingualResult && (
              <Card className="mt-8">
                <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-400">
                      🌍 MULTILINGUAL OCR
                    </span>

                    <h3 className="mt-3 text-xl font-bold">
                      Language-Aware Image Analysis
                    </h3>

                    <p className="mt-1 text-sm text-slate-500">
                      OCR text is language-detected and
                      translated before BERT analysis when
                      required.
                    </p>
                  </div>

                  <div className="rounded-xl border border-white/10 bg-slate-950/60 px-4 py-3 text-right">
                    <p className="text-xs text-slate-500">
                      Detected Language
                    </p>

                    <p className="mt-1 font-semibold text-cyan-400">
                      {imageMultilingualResult.language
                        ?.name ||
                        imageMultilingualResult.ocr
                          ?.ocr_language_name ||
                        "Unknown"}
                    </p>

                    {imageMultilingualResult
                      .language?.confidence !=
                      null && (
                      <p className="mt-1 text-xs text-slate-500">
                        Confidence{" "}
                        {(
                          imageMultilingualResult
                            .language.confidence * 100
                        ).toFixed(1)}
                        %
                      </p>
                    )}
                  </div>
                </div>

                <div className="grid gap-4 md:grid-cols-3">
                  <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                    <p className="text-xs text-slate-500">
                      OCR Language Hint
                    </p>

                    <p className="mt-2 text-lg font-semibold">
                      {imageMultilingualResult.ocr
                        ?.ocr_language_name ||
                        "Auto"}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                    <p className="text-xs text-slate-500">
                      Analysis Language
                    </p>

                    <p className="mt-2 text-lg font-semibold">
                      {imageMultilingualResult
                        .analysis_language
                        ?.name || "English"}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                    <p className="text-xs text-slate-500">
                      Translation
                    </p>

                    <p className="mt-2 text-lg font-semibold text-emerald-400">
                      {imageMultilingualResult.translation_applied
                        ? "Applied"
                        : "Not Required"}
                    </p>
                  </div>
                </div>

                {imageMultilingualResult.translation_applied && (
                  <div className="mt-4 rounded-2xl border border-emerald-400/10 bg-emerald-400/[0.03] p-5">
                    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-emerald-400">
                      English for BERT Analysis
                    </p>

                    <p className="whitespace-pre-wrap leading-7 text-slate-300">
                      {
                        imageMultilingualResult.translated_text
                      }
                    </p>
                  </div>
                )}
              </Card>
            )}

            {result && (
              <section className="mt-10 space-y-6">
                <div className="flex flex-col gap-2 border-b border-white/10 pb-5 sm:flex-row sm:items-end sm:justify-between">
                  <div>
                    <p className="text-xs font-medium uppercase tracking-[0.25em] text-cyan-400">
                      Analysis Result
                    </p>

                    <h3 className="mt-1 text-2xl font-bold">
                      TruthLens Assessment
                    </h3>
                  </div>

                  <p className="text-xs text-slate-500">
                    Prediction and verification are separate
                    signals
                  </p>
                </div>

                <div className="grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
                  <Card
                    className={
                      result.prediction ===
                      "FAKE"
                        ? "border-red-400/20 bg-red-400/[0.04]"
                        : "border-emerald-400/20 bg-emerald-400/[0.04]"
                    }
                  >
                    <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.2em] text-slate-500">
                          <span
                            className={`h-2 w-2 rounded-full ${
                              result.prediction ===
                              "FAKE"
                                ? "bg-red-400"
                                : "bg-emerald-400"
                            }`}
                          />
                          BERT assessment
                        </div>

                        <p
                          className={`mt-3 text-5xl font-black tracking-tight ${
                            result.prediction ===
                            "FAKE"
                              ? "text-red-400"
                              : "text-emerald-400"
                          }`}
                        >
                          {result.prediction}
                        </p>

                        <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
                          Model classification learned
                          from training data. It is not
                          independent factual verification.
                        </p>
                      </div>

                      <div className="shrink-0 rounded-2xl border border-white/10 bg-slate-950/50 px-6 py-5 text-center">
                        <p className="text-xs text-slate-500">
                          Confidence
                        </p>

                        <p className="mt-1 text-3xl font-bold text-cyan-400">
                          {(
                            result.confidence * 100
                          ).toFixed(2)}
                          %
                        </p>
                      </div>
                    </div>

                    <div className="mt-6 h-2 overflow-hidden rounded-full bg-white/10">
                      <div
                        className={`h-full rounded-full transition-all ${
                          result.prediction ===
                          "FAKE"
                            ? "bg-red-400"
                            : "bg-emerald-400"
                        }`}
                        style={{
                          width: `${safePercent(
                            result.confidence
                          )}%`,
                        }}
                      />
                    </div>
                  </Card>

                  <Card>
                    <p className="text-sm text-slate-500">
                      Class Probabilities
                    </p>

                    <div className="mt-4 space-y-4">
                      <div>
                        <div className="mb-1 flex justify-between text-sm">
                          <span className="text-red-400">
                            Fake
                          </span>

                          <span>
                            {(
                              result.fake_probability *
                              100
                            ).toFixed(2)}
                            %
                          </span>
                        </div>

                        <div className="h-2 rounded-full bg-white/10">
                          <div
                            className="h-full rounded-full bg-red-400"
                            style={{
                              width: `${safePercent(
                                result.fake_probability
                              )}%`,
                            }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="mb-1 flex justify-between text-sm">
                          <span className="text-emerald-400">
                            Real
                          </span>

                          <span>
                            {(
                              result.real_probability *
                              100
                            ).toFixed(2)}
                            %
                          </span>
                        </div>

                        <div className="h-2 rounded-full bg-white/10">
                          <div
                            className="h-full rounded-full bg-emerald-400"
                            style={{
                              width: `${safePercent(
                                result.real_probability
                              )}%`,
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </Card>
                </div>

                {credibility && (
                  <Card>
                    <div className="flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
                      <div>
                        <h3 className="text-xl font-bold">
                          Credibility Analysis
                        </h3>

                        <p className="mt-1 max-w-2xl text-sm text-slate-500">
                          An analytical indicator combining
                          model confidence, evidence coverage
                          and source diversity. It is not an
                          absolute truth score.
                        </p>
                      </div>

                      <div className="text-left md:text-right">
                        <p className="text-4xl font-bold text-cyan-400">
                          {credibility.score}/100
                        </p>

                        <p className="mt-1 text-xs text-slate-500">
                          {scoreLabel(
                            credibility.score
                          )}{" "}
                          analytical signal
                        </p>
                      </div>
                    </div>

                    <div className="mt-6 grid gap-4 sm:grid-cols-3">
                      <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                        <p className="text-xs text-slate-500">
                          Model confidence
                        </p>

                        <p className="mt-2 text-xl font-semibold">
                          {credibility.modelConfidence.toFixed(
                            1
                          )}
                          %
                        </p>
                      </div>

                      <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                        <p className="text-xs text-slate-500">
                          Evidence coverage
                        </p>

                        <p className="mt-2 text-xl font-semibold">
                          {credibility.evidenceCoverage.toFixed(
                            1
                          )}
                          %
                        </p>
                      </div>

                      <div className="rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                        <p className="text-xs text-slate-500">
                          Evidence sources
                        </p>

                        <p className="mt-2 text-xl font-semibold">
                          {credibility.sourceCount}
                        </p>
                      </div>
                    </div>
                  </Card>
                )}

                <Card title="Explainable AI">
                  <p className="mb-6 text-sm text-slate-500">
                    Tokens that contributed strongly to the
                    model prediction.
                  </p>

                  {result.important_tokens &&
                  result.important_tokens.length >
                    0 ? (
                    <div className="flex flex-wrap gap-3">
                      {result.important_tokens.map(
                        (item, index) => (
                          <div
                            key={`${item.token}-${index}`}
                            className="rounded-xl border border-cyan-400/20 bg-cyan-400/5 px-4 py-3"
                          >
                            <span className="font-medium text-cyan-300">
                              {item.token}
                            </span>

                            <span className="ml-3 text-xs text-slate-500">
                              {item.importance.toFixed(
                                4
                              )}
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  ) : (
                    <p className="text-sm text-slate-500">
                      No explanation tokens available.
                    </p>
                  )}
                </Card>

                <div className="grid gap-6 lg:grid-cols-2">
                  <Card title="Language & Manipulation Signals">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                        <div>
                          <p className="font-medium">
                            Clickbait / sensational wording
                          </p>

                          <p className="text-xs text-slate-500">
                            {heuristics.sensationalHits
                              .length
                              ? heuristics.sensationalHits.join(
                                  ", "
                                )
                              : "No obvious trigger words detected"}
                          </p>
                        </div>

                        <span className="text-cyan-400">
                          {heuristics.clickbaitScore}/100
                        </span>
                      </div>

                      <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                        <div>
                          <p className="font-medium">
                            Emotional / formatting signals
                          </p>

                          <p className="text-xs text-slate-500">
                            {heuristics.exclamations}{" "}
                            exclamation(s) ·{" "}
                            {(
                              heuristics.uppercaseRatio *
                              100
                            ).toFixed(1)}
                            % uppercase words
                          </p>
                        </div>

                        <span className="text-amber-400">
                          {heuristics.styleScore}/100
                        </span>
                      </div>

                      <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/50 p-4">
                        <div>
                          <p className="font-medium">
                            AI-style linguistic signals
                          </p>

                          <p className="text-xs text-slate-500">
                            Pattern-based indicator, not an
                            AI-generated-content verdict
                          </p>
                        </div>

                        <span className="text-violet-400">
                          {heuristics.aiSignalScore}/100
                        </span>
                      </div>
                    </div>
                  </Card>

                  <Card title="Claim Verification">
                    {verification ? (
                      <>
                        <div className="mb-5 flex items-center justify-between">
                          <span className="text-sm text-slate-400">
                            {verification.claims_checked ??
                              verification.claims?.length ??
                              0}{" "}
                            claims checked
                          </span>

                          <span className="rounded-full bg-cyan-400/10 px-3 py-1 text-xs text-cyan-400">
                            {verification.overall_assessment ||
                              (verification.status ===
                              "COMPLETED"
                                ? "Completed"
                                : "Analysis")}
                          </span>
                        </div>

                        {[
                          "PENDING",
                          "PROCESSING",
                          "RUNNING",
                        ].includes(
                          String(
                            verification.status || ""
                          ).toUpperCase()
                        ) ? (
                          <div className="rounded-2xl border border-cyan-400/15 bg-cyan-400/[0.04] p-5">
                            <div className="flex items-start gap-4">
                              <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-cyan-400/20 bg-cyan-400/10 text-cyan-400">
                                <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-cyan-400" />
                              </div>

                              <div>
                                <p className="font-semibold text-slate-200">
                                  Evidence verification in progress
                                </p>

                                <p className="mt-1 text-sm leading-6 text-slate-500">
                                  TruthLens is independently
                                  checking extracted claims
                                  against available evidence.
                                  Your BERT result is already
                                  available.
                                </p>

                                <p className="mt-3 text-xs text-cyan-400">
                                  Checking in the background…
                                </p>
                              </div>
                            </div>
                          </div>
                        ) : verification.claims &&
                          verification.claims.length >
                            0 ? (
                          <div className="space-y-4">
                            {verification.claims.map(
                              (claim, index) => (
                                <div
                                  key={index}
                                  className="rounded-2xl border border-white/10 bg-slate-950/50 p-4"
                                >
                                  <p className="text-sm leading-6 text-slate-300">
                                    {claim.claim}
                                  </p>

                                  <div className="mt-3 flex flex-wrap gap-2 text-xs">
                                    <span className="rounded-full bg-white/5 px-3 py-1 text-slate-400">
                                      {claim.assessment ||
                                        "INSUFFICIENT_EVIDENCE"}
                                    </span>

                                    {typeof claim.support_score ===
                                      "number" && (
                                      <span className="rounded-full bg-emerald-400/10 px-3 py-1 text-emerald-400">
                                        Support{" "}
                                        {(
                                          claim.support_score *
                                          100
                                        ).toFixed(1)}
                                        %
                                      </span>
                                    )}

                                    {typeof claim.refute_score ===
                                      "number" && (
                                      <span className="rounded-full bg-red-400/10 px-3 py-1 text-red-400">
                                        Refute{" "}
                                        {(
                                          claim.refute_score *
                                          100
                                        ).toFixed(1)}
                                        %
                                      </span>
                                    )}
                                  </div>

                                  {claim.evidence &&
                                    claim.evidence.length >
                                      0 && (
                                      <div className="mt-4 space-y-2">
                                        {claim.evidence
                                          .slice(0, 3)
                                          .map(
                                            (
                                              evidence,
                                              eIndex
                                            ) => (
                                              <div
                                                key={
                                                  eIndex
                                                }
                                                className="rounded-xl border border-white/10 p-3"
                                              >
                                                <p className="text-xs font-medium text-slate-300">
                                                  {evidence.title ||
                                                    evidence.domain ||
                                                    "Evidence"}
                                                </p>

                                                {evidence.text && (
                                                  <p className="mt-1 text-xs leading-5 text-slate-500">
                                                    {
                                                      evidence.text
                                                    }
                                                  </p>
                                                )}

                                                {evidence.url && (
                                                  <a
                                                    href={
                                                      evidence.url
                                                    }
                                                    target="_blank"
                                                    rel="noreferrer"
                                                    className="mt-2 block break-all text-xs text-cyan-400 hover:underline"
                                                  >
                                                    {
                                                      evidence.url
                                                    }
                                                  </a>
                                                )}
                                              </div>
                                            )
                                          )}
                                      </div>
                                    )}
                                </div>
                              )
                            )}
                          </div>
                        ) : (
                          <p className="text-sm text-slate-500">
                            No claims could be independently
                            assessed from the available evidence.
                          </p>
                        )}

                        {verification.note && (
                          <p className="mt-5 text-xs leading-5 text-slate-500">
                            {verification.note}
                          </p>
                        )}
                      </>
                    ) : (
                      <p className="text-sm leading-6 text-slate-500">
                        Verification data was not returned.
                        The BERT result above remains a
                        classification signal, not independent
                        fact verification.
                      </p>
                    )}
                  </Card>
                </div>

                <Card title="Analysis Pipeline">
                  <div className="flex flex-wrap items-center gap-3">
                    {pipeline.map((step, index) => (
                      <div
                        key={`${step}-${index}`}
                        className="flex items-center gap-3"
                      >
                        <div className="rounded-xl border border-white/10 bg-slate-900 px-4 py-3 text-sm text-slate-300">
                          {step}
                        </div>

                        {index <
                          pipeline.length - 1 && (
                          <span className="text-cyan-400">
                            →
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </Card>

                <div className="rounded-2xl border border-amber-400/10 bg-amber-400/[0.03] px-5 py-4">
                  <p className="text-xs leading-6 text-slate-500">
                    <span className="font-medium text-amber-400">
                      Important:
                    </span>{" "}
                    The BERT prediction is based on patterns
                    learned from its training dataset.
                    Verification and evidence retrieval are
                    separate layers. Linguistic/AI-style
                    indicators are analytical signals and
                    should not be interpreted as proof that
                    content is fake or AI-generated.
                  </p>
                </div>
              </section>
            )}
          </>
        )}
      </main>

      <footer className="border-t border-white/10 py-6 text-center text-xs text-slate-600">
        TruthLens AI • Explainable Fake News Detection System
      </footer>
    </div>
  );
}

export default App;