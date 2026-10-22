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



const SessionSchema = new mongoose.Schema(
  {
    userId: { type: String, default: null },
    fastapiId: { type: String, required: true },
    pendingQuestion: { type: String, default: "" },
    role: { type: String, required: true },
    experienceLevel: { type: String, default: "junior" },
    timeframe: { type: String, default: "" },
    cv: { type: String, default: "" },
    mode: { type: String, default: "text" },
    minutes: { type: Number, default: 10 },
    company: { type: String, default: "" },
    status: { type: String, default: "in_progress" },
    transcript: [
      {
        question: String,
        userAnswer: String,
        followUp: Boolean,
      },
    ],
    overallScore: { type: Number, default: null },
    scores: { type: Object, default: {} },
    feedback: { type: Array, default: [] },
    evaluation: { type: Object, default: {} },
    report: { type: Object, default: {} },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Session", SessionSchema);