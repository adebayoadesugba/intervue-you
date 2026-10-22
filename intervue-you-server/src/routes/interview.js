const express = require("express");
const router = express.Router();
const Session = require("../models/Session");
const OpenAI = require("openai");
const jwt = require("jsonwebtoken");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const FASTAPI_BASE_URL = process.env.FASTAPI_URL || "http://127.0.0.1:8000";

function extractUserId(req) {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith("Bearer ")) {
    try {
      const token = authHeader.split(" ")[1];
      if (token && process.env.JWT_SECRET) {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        return decoded.id || decoded._id || decoded.userId || null;
      }
    } catch (e) {
      return null;
    }
  }
  return null;
}

// Helper to normalize frontend inputs into strict FastAPI StartRequest schema
function buildFastApiPayload(reqBody) {
  const { role, category, mode, minutes, experienceLevel, stack, cv } = reqBody;

  const targetCategory = (category || role || "frontend").toLowerCase().trim();
  const targetDifficulty = (experienceLevel || "junior").toLowerCase().trim();
  const targetStack = Array.isArray(stack) && stack.length > 0 ? stack : ["core"];
  
  const targetMode = mode === "audio" ? "voice" : (mode || "text");
  const numMinutes = Math.max(1, Math.min(60, Number(minutes) || 10));

  return {
    category: targetCategory,
    difficulty: targetDifficulty,
    stack: targetStack,
    mode: targetMode,
    time_limit_minutes: numMinutes,
    cv_text: cv && cv.trim() ? cv.trim() : null,
  };
}

// ==========================================
// 1. REAL-TIME STREAMING: CREATE SESSION
// ==========================================
const createSessionStreamHandler = async (req, res) => {
  try {
    const userId = extractUserId(req);
    const fastApiPayload = buildFastApiPayload(req.body);

    const fastApiRes = await fetch(`${FASTAPI_BASE_URL}/interview/start/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fastApiPayload),
    });

    if (!fastApiRes.ok) {
      const errorText = await fastApiRes.text();
      return res.status(fastApiRes.status).json({ error: errorText });
    }

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    let fastapiSessionId = null;
    let completeQuestionText = "";

    const reader = fastApiRes.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      res.write(chunk);

      const lines = chunk.split("\n");
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === "session_created") {
              fastapiSessionId = data.session_id;
            } else if (data.type === "token") {
              completeQuestionText += data.content;
            } else if (data.type === "done") {
              const sessionData = {
                userId,
                fastapiId: fastapiSessionId || data.session_id,
                pendingQuestion: data.question || completeQuestionText,
                role: fastApiPayload.category,
                experienceLevel: fastApiPayload.difficulty,
                timeframe: req.body.timeframe || "",
                cv: req.body.cv || "",
                mode: req.body.mode || "text",
                minutes: fastApiPayload.time_limit_minutes,
                company: req.body.company || "",
                status: "in_progress",
                transcript: [],
              };
              const session = new Session(sessionData);
              await session.save();
            }
          } catch (e) {
            // Ignore partial SSE parses
          }
        }
      }
    }

    res.end();
  } catch (err) {
    console.error("❌ Express POST /session/stream error:", err);
    if (!res.headersSent) {
      res.status(500).json({ error: "Failed to stream session start" });
    }
  }
};
router.post(["/session/stream", "/interview/session/stream"], createSessionStreamHandler);

// ==========================================
// 2. REAL-TIME STREAMING: NEXT QUESTION
// ==========================================
const nextQuestionStreamHandler = async (req, res) => {
  try {
    const session = await Session.findById(req.params.id);
    if (!session) return res.status(404).json({ error: "Session not found" });

    const incomingTurns = req.body?.session?.turns || [];
    if (incomingTurns.length === 0) {
      return res.json({ q: session.pendingQuestion, followUp: false });
    }

    const lastTurn = incomingTurns[incomingTurns.length - 1];

    session.transcript = incomingTurns.map((t) => ({
      question: t.q,
      userAnswer: t.a,
      followUp: Boolean(t.followUp),
    }));

    const fastApiRes = await fetch(`${FASTAPI_BASE_URL}/interview/${session.fastapiId}/answer/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer: lastTurn.a }),
    });

    if (!fastApiRes.ok) {
      const errorText = await fastApiRes.text();
      return res.status(fastApiRes.status).json({ error: errorText });
    }

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    const reader = fastApiRes.body.getReader();
    const decoder = new TextDecoder();
    let nextQuestionText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      res.write(chunk);

      const lines = chunk.split("\n");
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === "token") {
              nextQuestionText += data.content;
            } else if (data.type === "ended") {
              session.status = "completed";
              await session.save();
            } else if (data.type === "done") {
              session.pendingQuestion = data.next_question || nextQuestionText;
              await session.save();
            }
          } catch (e) {
            // Ignore partial SSE parsing issues
          }
        }
      }
    }

    res.end();
  } catch (err) {
    console.error("❌ Express next-question stream error:", err);
    if (!res.headersSent) {
      res.status(500).json({ error: "Failed to stream next question" });
    }
  }
};
router.post(["/session/:id/next-question/stream", "/interview/session/:id/next-question/stream"], nextQuestionStreamHandler);

// ==========================================
// 3. NON-STREAMING CREATE SESSION
// ==========================================
const createSessionHandler = async (req, res) => {
  try {
    const userId = extractUserId(req);
    const fastApiPayload = buildFastApiPayload(req.body);

    const fastApiRes = await fetch(`${FASTAPI_BASE_URL}/interview/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(fastApiPayload),
    });

    if (!fastApiRes.ok) {
      const errorText = await fastApiRes.text();
      return res.status(fastApiRes.status).json({
        error: `FastAPI rejected session initialization`,
        details: errorText,
      });
    }

    const fastApiData = await fastApiRes.json();

    const sessionData = {
      userId,
      fastapiId: fastApiData.session_id,
      pendingQuestion: fastApiData.question,
      role: fastApiPayload.category,
      experienceLevel: fastApiPayload.difficulty,
      timeframe: req.body.timeframe || "",
      cv: req.body.cv || "",
      mode: req.body.mode || "text",
      minutes: fastApiPayload.time_limit_minutes,
      company: req.body.company || "",
      status: "in_progress",
      transcript: [],
    };

    const session = new Session(sessionData);
    await session.save();

    return res.status(201).json({ session });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
};
router.post(["/session", "/interview/session"], createSessionHandler);

// ==========================================
// 4. GET SESSION LIST & INDIVIDUAL SESSION
// ==========================================
const getAllSessionsHandler = async (req, res) => {
  try {
    const userId = extractUserId(req);
    const filter = userId ? { userId } : {};
    const sessions = await Session.find(filter).sort({ createdAt: -1 }).limit(20);
    return res.json({ sessions });
  } catch (err) {
    console.error("Error fetching sessions:", err);
    return res.status(500).json({ error: "Failed to fetch session history" });
  }
};
router.get(["/sessions", "/session", "/interview/sessions"], getAllSessionsHandler);

const getSessionHandler = async (req, res) => {
  try {
    const session = await Session.findById(req.params.id);
    if (!session) return res.status(404).json({ error: "Session not found" });
    return res.json({ session });
  } catch (err) {
    return res.status(500).json({ error: "Failed to fetch session" });
  }
};
router.get(["/session/:id", "/sessions/:id", "/interview/session/:id"], getSessionHandler);

// ==========================================
// 5. NEXT QUESTION (NON-STREAMING)
// ==========================================
const nextQuestionHandler = async (req, res) => {
  try {
    const session = await Session.findById(req.params.id);
    if (!session) return res.status(404).json({ error: "Session not found" });

    const incomingTurns = req.body?.session?.turns || [];
    if (incomingTurns.length === 0) {
      return res.json({ q: session.pendingQuestion, followUp: false });
    }

    const lastTurn = incomingTurns[incomingTurns.length - 1];
    
    session.transcript = incomingTurns.map((t) => ({
      question: t.q,
      userAnswer: t.a,
      followUp: Boolean(t.followUp),
    }));

    const fastApiRes = await fetch(`${FASTAPI_BASE_URL}/interview/${session.fastapiId}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer: lastTurn.a }),
    });

    const fastApiData = await fastApiRes.json();
    const isFollowUp = Boolean(fastApiData.is_follow_up || fastApiData.isFollowUp || fastApiData.followUp);

    if (fastApiData.ended) {
      session.status = "completed";
      await session.save();
      return res.json({ 
        q: "That concludes our interview session. Submitting your responses for evaluation.", 
        followUp: false,
        ended: true,
        reason: fastApiData.reason || "completed"
      });
    }

    session.pendingQuestion = fastApiData.next_question;
    await session.save();

    return res.json({ 
      q: fastApiData.next_question, 
      followUp: isFollowUp,
      ended: false 
    });
  } catch (err) {
    return res.status(500).json({ error: "Failed to retrieve RAG question" });
  }
};
router.post(["/session/:id/next-question", "/interview/session/:id/next-question"], nextQuestionHandler);

// ==========================================
// 6. EVALUATION ROUTE (PYTHON REPORT)
// ==========================================
// ==========================================
// EVALUATION ROUTE (FETCHES PYTHON REPORT)
// ==========================================
const evaluateHandler = async (req, res) => {
  try {
    const session = await Session.findById(req.params.id);
    if (!session) return res.status(404).json({ error: "Session not found" });

    const incomingTurns = req.body?.transcript;
    if (Array.isArray(incomingTurns) && incomingTurns.length > 0) {
      session.transcript = incomingTurns.map((t) => ({
        question: t.q,
        userAnswer: t.a,
        followUp: Boolean(t.followUp),
      }));
      await session.save();
    }

    // Call FastAPI RAG endpoint (report_agent.py generation)
    const fastApiRes = await fetch(`${FASTAPI_BASE_URL}/interview/${session.fastapiId}/report`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    if (!fastApiRes.ok) {
      const errorText = await fastApiRes.text();
      console.error("❌ FastAPI /report fetch error:", errorText);
      return res.status(fastApiRes.status).json({ error: "Python RAG evaluation failed", details: errorText });
    }

    const fastApiData = await fastApiRes.json();
    
    // Safely unwrap report whether returned at root or under .report
    const reportData = fastApiData.report || fastApiData;
    const comps = reportData.competencies || {};

    const calculatedScores = {
      technical: comps.technical_depth?.score ?? fastApiData.scores?.technical ?? null,
      communication: comps.communication?.score ?? fastApiData.scores?.communication ?? null,
      problemSolving: comps.problem_solving?.score ?? fastApiData.scores?.problemSolving ?? null,
      teamwork: comps.teamwork?.score ?? fastApiData.scores?.teamwork ?? null,
    };

    const overallScore = reportData.overall_score ?? fastApiData.overallScore ?? 0;

    // Save normalized objects in Mongo
    session.report = reportData;
    session.overallScore = overallScore;
    session.scores = calculatedScores;
    session.evaluation = {
      overallScore,
      scores: calculatedScores,
      report: reportData,
      feedback: fastApiData.feedback || reportData.feedback || [],
    };
    session.status = "completed";

    await session.save();

    return res.json({ session });
  } catch (err) {
    console.error("❌ Express /evaluate error:", err);
    return res.status(500).json({ error: "Failed to retrieve Python scorecard" });
  }
};
router.post(["/session/:id/evaluate", "/interview/session/:id/evaluate"], evaluateHandler);

// ==========================================
// 7. COACH ROUTE
// ==========================================
const coachHandler = async (req, res) => {
  try {
    const { message, text, chatHistory = [] } = req.body;
    const userMessage = message || text;
    if (!userMessage) return res.status(400).json({ error: "A message string is required" });

    const session = await Session.findById(req.params.id);
    if (!session) return res.status(404).json({ error: "Session not found" });

    const report = session.report || {};
    const comps = report.competencies || {};

    const systemContext = `
You are Intervue You's expert AI Executive Interview Coach.
Coaching candidate on their interview for "${session.role}".
Python Calculated Scores: Overall ${session.overallScore}%. Tech: ${comps.technical_depth?.score ?? "N/A"}. Comm: ${comps.communication?.score ?? "N/A"}. Problem Solving: ${comps.problem_solving?.score ?? "N/A"}. Teamwork: ${comps.teamwork?.score ?? "N/A"}.
Executive Summary: ${report.overall_summary || "N/A"}
Provide actionable coaching advice based strictly on their transcript and Python report.`;

    const formattedHistory = chatHistory
      .map((c) => ({
        role: c.sender === "user" || c.me ? "user" : "assistant",
        content: c.text || c.t || "",
      }))
      .filter((c) => c.content.trim().length > 0);

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "system", content: systemContext }, ...formattedHistory, { role: "user", content: userMessage }],
      temperature: 0.7,
      max_tokens: 500,
    });

    return res.json({ reply: response.choices[0].message.content });
  } catch (err) {
    return res.status(500).json({ error: "Coach failed to respond" });
  }
};
router.post(["/session/:id/coach", "/interview/session/:id/coach"], coachHandler);

module.exports = router;