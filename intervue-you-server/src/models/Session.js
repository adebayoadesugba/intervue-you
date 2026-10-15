const mongoose = require("mongoose");

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
    R: { type: Boolean, default: false },
  },
});

const sessionSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: false },
    fastapiId: { type: String }, // Links to the Python session ID
    pendingQuestion: { type: String }, // Holds the RAG question awaiting an answer
    role: { type: String, required: true },
    experienceLevel: { type: String, default: "junior" },
    timeframe: { type: String, default: "" },
    cv: { type: String, default: "" },
    mode: { type: String, required: true, default: "text" },
    minutes: { type: Number, default: 10 },
    company: { type: String, default: "" },
    status: { type: String, enum: ["in_progress", "completed"], default: "in_progress" },
    overallScore: { type: Number },
    scores: {
      technical: { type: Number },
      communication: { type: Number },
      problemSolving: { type: Number },
      teamwork: { type: Number },
    },
    feedback: [feedbackSchema],
    transcript: { type: Array, default: [] },
    evaluation: { type: Object },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Session", sessionSchema);