const mongoose = require("mongoose");

const profileSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, unique: true },
    role: { type: String, required: true },
    level: { type: String, required: true },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Profile", profileSchema);