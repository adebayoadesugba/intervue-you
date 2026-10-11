const express = require("express");
const router = express.Router();
const Session = require("../models/Session");
const OpenAI = require("openai");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// GET SESSION BY ID (Handles both /session/:id and /sessions/:id)
const getSessionHandler = async (req, res) => {
  try {
    const session = await Session.findById(req.params.id);
    if (!session) return res.status(404).json({ error: "Session not found" });
    return res.json({ session });
  } catch (err) {
    console.error("Fetch session error:", err);
    return res.status(500).json({ error: "Failed to fetch session" });
  }
};

router.get("/session/:id", getSessionHandler);
router.get("/sessions/:id", getSessionHandler);

// 1. GENERATE AI EVALUATION & SCORES FOR INTERVIEW
router.post("/session/:id/evaluate", async (req, res) => {
  try {
    const session = await Session.findById(req.params.id);
    if (!session) return res.status(404).json({ error: "Session not found" });

    // Normalize transcript entries (supports 'question'/'q' and 'userAnswer'/'a'/'text')
    const rawTranscript = session.transcript || session.turns || [];
    if (!rawTranscript.length) {
      return res.status(400).json({ error: "Cannot evaluate a session with an empty transcript" });
    }

    const normalizedTranscript = rawTranscript.map((item, idx) => ({
      index: idx + 1,
      q: item.question || item.q || `Question ${idx + 1}`,
      a: item.userAnswer || item.a || item.text || "(No answer provided)",
    }));

    const transcriptText = normalizedTranscript
      .map((item) => `Q${item.index}: ${item.q}\nA: ${item.a}`)
      .join("\n\n");

    const systemPrompt = `
You are an elite corporate technical and behavioral interview evaluator.
Analyze the provided interview transcript for a candidate targeting the role of "${session.role || "Software Engineer"}" at level "${session.level || "Mid-level"}".
Do NOT invent any context or facts; base your evaluation strictly on the transcript provided.

EVALUATION RULES:
1. Base all scores strictly on the provided transcript. Do NOT invent context or facts.
2. Penalize short, vague, incomplete, or weak responses (e.g., "shook hands", "we agreed", "idk").
3. Reward structured STAR responses (Situation, Task, Action, Result) with clear actions and quantifiable impact.
4. For every question in the transcript, evaluate whether Situation (S), Task (T), Action (A), and Result (R) were present in the user's answer.

Return ONLY a valid JSON object matching this exact schema:
{
  "overallScore": number (0-100 integer),
  "readinessLabel": "Ready to interview" | "Building confidence" | "Needs practice",
  "scores": {
    "technical": number (0-100 integer),
    "communication": number (0-100 integer),
    "problemSolving": number (0-100 integer),
    "teamwork": number (0-100 integer)
  },
  "starAssessment": {
    "situation": "Summary of how well situation context was established",
    "task": "Summary of how clearly responsibilities were articulated",
    "action": "Summary of specific individual steps highlighted",
    "result": "Summary of measurable metrics and outcomes provided"
  },
  "feedback": [
    {
      "q": "Exact or summarized question string",
      "a": "Candidate's response string",
      "score": number (0-100 integer score for this specific answer),
      "note": "1-sentence concise critique on why this score was awarded",
      "star": {
        "S": boolean,
        "T": boolean,
        "A": boolean,
        "R": boolean
      },
      "better": "1-2 sentence recommendation showing a stronger alternative response using STAR format"
    }
  ],
  "strengths": ["string", "string"],
  "improvements": ["string", "string"]
}
`;

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: `Interview Transcript:\n\n${transcriptText}` },
      ],
      response_format: { type: "json_object" },
      temperature: 0.2,
    });

    const evaluation = JSON.parse(response.choices[0].message.content || "{}");

    // Save evaluation to MongoDB session
    session.evaluation = evaluation;
    session.overallScore = evaluation.overallScore;
    session.scores = evaluation.scores;
    session.feedback = evaluation.feedback;
    session.status = "completed";
    await session.save();

    return res.json({ session });
  } catch (err) {
    console.error("Evaluation error:", err);
    return res.status(500).json({ error: "Failed to generate evaluation" });
  }
});

// 2. "ASK YOUR COACH" AI CHAT ENDPOINT
router.post("/session/:id/coach", async (req, res) => {
  try {
    const { message, text, chatHistory = [] } = req.body;
    const userMessage = message || text;

    if (!userMessage) {
      return res.status(400).json({ error: "A message string is required" });
    }

    const session = await Session.findById(req.params.id);
    if (!session) return res.status(404).json({ error: "Session not found" });

    const evaluation = session.evaluation || {};
    const scores = evaluation.scores || session.scores || {};
    const star = evaluation.starAssessment || {};
    const feedbackList = evaluation.feedback || session.feedback || [];

    const rawTranscript = session.transcript || session.turns || [];
    const transcriptSummary = rawTranscript
      .map((t, i) => `Q${i + 1}: ${t.question || t.q}\nA: ${t.userAnswer || t.a || t.text}`)
      .join("\n");

    const systemContext = `
You are Intervue You's expert AI Executive Interview Coach.
You are directly coaching ${session.userName || "the candidate"} on their interview for the position of "${session.role || "Target Role"}".
Do not invent any context or facts; base your coaching strictly on the provided transcript, scores, and feedback.
Do not provide generic advice; refer specifically to their actual questions, weak scores, or missing STAR components when answering.
Do not answer questions that are not related to the candidate's performance in this specific interview session or question that are not related to the interview transcript provided.
CANDIDATE PERFORMANCE DATA:
- Overall Score: ${evaluation.overallScore || session.overallScore || "N/A"}%
- Technical: ${scores.technical || "N/A"}/100
- Communication: ${scores.communication || "N/A"}/100
- Problem Solving: ${scores.problemSolving || "N/A"}/100
- Teamwork: ${scores.teamwork || "N/A"}/100

STAR SUMMARY:
- Situation: ${star.situation || "N/A"}
- Task: ${star.task || "N/A"}
- Action: ${star.action || "N/A"}
- Result: ${star.result || "N/A"}

QUESTION FEEDBACK:
${JSON.stringify(feedbackList, null, 2)}

FULL TRANSCRIPT:
${transcriptSummary}

INSTRUCTIONS:
1. Provide concise, direct, supportive, and actionable coaching advice.
2. Refer specifically to their actual questions, weak scores, or missing STAR components when answering.
3. Keep answers concise (2 to 4 paragraphs max) so they are easy to read in a side-chat interface.
`;

    // Map history supporting both { sender: 'user'|'coach', text: '' } and { me: true|false, t: '' } formats
    const formattedHistory = chatHistory.map((c) => {
      const isUser = c.sender === "user" || c.me === true;
      return {
        role: isUser ? "user" : "assistant",
        content: c.text || c.t || "",
      };
    }).filter((item) => item.content.trim().length > 0);

    const messages = [
      { role: "system", content: systemContext },
      ...formattedHistory,
      { role: "user", content: userMessage },
    ];

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      temperature: 0.7,
      max_tokens: 500,
    });

    const reply = response.choices[0].message.content;
    return res.json({ reply });
  } catch (err) {
    console.error("Coach chat error:", err);
    return res.status(500).json({ error: "Coach failed to respond" });
  }
});

module.exports = router;