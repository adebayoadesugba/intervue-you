const mongoose = require("mongoose");

// Detailed feedback schema to preserve STAR breakdown
const feedbackSchema = new mongoose.Schema({
  q: { type: String, required: true },
  a: { type: String, required: true },
  score: { type: Number, required: true },
  note: { type: String },
  better: { type: String },
  star: {
    S: { type: Boolean, default: false },
    T: { type: Boolean, default: false },
    A: { type: Boolean, default: false },
    R: { type: Boolean, default: false }
  }
});

const sessionSchema = new mongoose.Schema(
  {
    userId: { 
      type: mongoose.Schema.Types.ObjectId, 
      ref: "User", 
      required: true 
    },
    role: { type: String, required: true },
    experienceLevel: { type: String, default: "Mid-level" },
    mode: { type: String, required: true },
    overallScore: { type: Number, required: true },
    scores: {
      technical: Number,
      communication: Number,
      problemSolving: Number,
      teamwork: Number
    },
    feedback: [feedbackSchema], // <--- Preserves full STAR array per question
    transcript: { type: Array, default: [] }
  },
  { timestamps: true }
);

module.exports = mongoose.model("Session", sessionSchema);