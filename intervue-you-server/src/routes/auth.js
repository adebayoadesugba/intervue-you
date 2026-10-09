const express = require("express");
const crypto = require("crypto");
const bcrypt = require("bcryptjs");
const nodemailer = require("nodemailer");
const jwt = require("jsonwebtoken");
const mongoose = require("mongoose");
const User = require("../models/User");
const { OAuth2Client } = require("google-auth-library");

const router = express.Router();
const client = new OAuth2Client(process.env.GOOGLE_CLIENT_ID);

// Safe helper to retrieve profile if a Profile model is registered in Mongoose
async function getProfileSafely(userId) {
  try {
    if (mongoose.models.Profile) {
      return await mongoose.models.Profile.findOne({ userId });
    }
  } catch (_) {
    // Return null if model/collection doesn't exist yet
  }
  return null;
}

// Nodemailer SMTP Transporter
const transporter = nodemailer.createTransport({
  service: "Gmail",
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS,
  },
});

// POST /api/auth/register
router.post("/register", async (req, res) => {
  try {
    const { name, email, password } = req.body;

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ error: "Email is already in use." });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const user = new User({ name, email, password: hashedPassword });
    await user.save();

    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET || "your_jwt_secret", { expiresIn: "7d" });

    res.status(201).json({
      token,
      user: { id: user._id, name: user.name, email: user.email },
      profile: null,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal server error during registration." });
  }
});

// POST /api/auth/login
router.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ error: "Invalid email or password." });
    }

    const isMatch = await bcrypt.compare(password, user.password || "");
    if (!isMatch) {
      return res.status(400).json({ error: "Invalid email or password." });
    }

    const userProfile = await getProfileSafely(user._id);
    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET || "your_jwt_secret", { expiresIn: "7d" });

    res.status(200).json({
      token,
      user: { id: user._id, name: user.name, email: user.email },
      profile: userProfile,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal server error during login." });
  }
});

// POST /api/auth/google
router.post("/google", async (req, res) => {
  try {
    const { token } = req.body;

    if (!token) {
      return res.status(400).json({ message: "Google token is required" });
    }

    const googleRes = await fetch("https://www.googleapis.com/oauth2/v3/userinfo", {
      headers: { Authorization: `Bearer ${token}` },
    });

    if (!googleRes.ok) {
      return res.status(400).json({ message: "Failed to verify Google token" });
    }

    const payload = await googleRes.json();
    const { email, name, sub: googleId } = payload;

    if (!email) {
      return res.status(400).json({ message: "Email not provided by Google" });
    }

    let user = await User.findOne({ email });

    if (!user) {
      const randomPassword = crypto.randomBytes(16).toString("hex");
      user = await User.create({
        name: name || email.split("@")[0],
        email,
        password: randomPassword,
        googleId,
        isGoogleAuth: true,
      });
    }

    const userProfile = await getProfileSafely(user._id);
    const appToken = jwt.sign(
      { userId: user._id, email: user.email },
      process.env.JWT_SECRET || "your_jwt_secret",
      { expiresIn: "7d" }
    );

    return res.json({
      token: appToken,
      user: { id: user._id, name: user.name, email: user.email },
      profile: userProfile,
    });
  } catch (error) {
    console.error("Google auth server error:", error);
    return res.status(500).json({ message: "Internal server error during Google login" });
  }
});

// POST /api/auth/forgot-password
router.post("/forgot-password", async (req, res) => {
  const { email } = req.body;
  try {
    const user = await User.findOne({ email });
    if (!user) {
      return res.json({ message: "If an account exists, a reset link has been sent." });
    }

    const resetToken = crypto.randomBytes(32).toString("hex");
    const hashedToken = crypto.createHash("sha256").update(resetToken).digest("hex");

    user.resetPasswordToken = hashedToken;
    user.resetPasswordExpires = Date.now() + 15 * 60 * 1000;
    await user.save();

    const resetUrl = `http://localhost:5173/reset-password?token=${resetToken}&id=${user._id}`;

    await transporter.sendMail({
      to: user.email,
      subject: "Password Reset — Intervue You",
      html: `<p>You requested a password reset. Click the link below to set a new password (valid for 15 minutes):</p>
             <a href="${resetUrl}">${resetUrl}</a>`,
    });

    res.json({ message: "Password reset link sent to your email." });
  } catch (err) {
    console.error("Forgot password error:", err);
    res.status(500).json({ message: "Error sending reset email." });
  }
});

// POST /api/auth/reset-password
router.post("/reset-password", async (req, res) => {
  const { token, userId, newPassword } = req.body;
  try {
    const hashedToken = crypto.createHash("sha256").update(token).digest("hex");

    const user = await User.findOne({
      _id: userId,
      resetPasswordToken: hashedToken,
      resetPasswordExpires: { $gt: Date.now() },
    });

    if (!user) {
      return res.status(400).json({ message: "Token is invalid or has expired." });
    }

    user.password = await bcrypt.hash(newPassword, 10);
    user.resetPasswordToken = undefined;
    user.resetPasswordExpires = undefined;
    await user.save();

    res.json({ message: "Password reset successful! You can now log in." });
  } catch (err) {
    console.error("Reset password error:", err);
    res.status(500).json({ message: "Error resetting password." });
  }
});

module.exports = router;