const express = require("express");
const jwt = require("jsonwebtoken");
const Session = require("../models/Session");
const router = express.Router();

// Middleware to verify JWT token from authorization header
const authMiddleware = (req, res, next) => {
  const authHeader = req.header("Authorization");
  if (!authHeader) return res.status(401).json({ error: "Access denied. No token provided." });

  const token = authHeader.split(" ")[1];
  try {
    const verified = jwt.verify(token, process.env.JWT_SECRET);
    req.user = verified; // Contains { userId: "..." }
    next();
  } catch (err) {
    res.status(400).json({ error: "Invalid token." });
  }
};

// POST /api/sessions - Save finished interview session
router.post("/", authMiddleware, async (req, res) => {
  try {
    const session = new Session({
      userId: req.user.userId,
      role: req.body.role,
      experienceLevel: req.body.experienceLevel,
      mode: req.body.mode,
      overallScore: req.body.overallScore,
      feedback: req.body.feedback,
      transcript: req.body.transcript
    });

    const savedSession = await session.save();
    res.status(201).json(savedSession);
  } catch (error) {
    res.status(500).json({ error: "Failed to save session to database." });
  }
});

// GET /api/sessions - Retrieve user's session history for dashboard
router.get("/", authMiddleware, async (req, res) => {
  try {
    const userSessions = await Session.find({ userId: req.user.userId }).sort({ createdAt: -1 });
    res.status(200).json(userSessions);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch session history." });
  }
});

// GET /api/sessions/:id - Fetch a single session by MongoDB ID
router.get("/:id", authMiddleware, async (req, res) => {
  try {
    const session = await Session.findOne({ _id: req.params.id, userId: req.user.userId });
    if (!session) return res.status(404).json({ error: "Session not found." });
    res.status(200).json(session);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch session." });
  }
});

module.exports = router;