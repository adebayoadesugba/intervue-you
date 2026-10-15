const express = require("express");
const crypto = require("crypto");
const bcrypt = require("bcryptjs");
const nodemailer = require("nodemailer");
const jwt = require("jsonwebtoken");
const User = require("../models/User");
const Profile = require("../models/Profile");
const { OAuth2Client } = require("google-auth-library");

const router = express.Router();
const client = new OAuth2Client(process.env.GOOGLE_CLIENT_ID);

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
      console.warn("⚠️ Token verification failed in extractUserId:", e.message);
      return null;
    }
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

// Middleware to verify JWT token
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];
  if (!token) return res.status(401).json({ error: "Access token required" });

  jwt.verify(token, process.env.JWT_SECRET || "your_jwt_secret", (err, user) => {
    if (err) return res.status(403).json({ error: "Invalid or expired token" });
    req.user = user;
    next();
  });
};

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
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        hasCompletedOnboarding: Boolean(user.hasCompletedOnboarding),
      },
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

    // Retrieve onboarding profile from MongoDB
    const userProfile = await Profile.findOne({ userId: user._id });
    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET || "your_jwt_secret", { expiresIn: "7d" });

    res.status(200).json({
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        hasCompletedOnboarding: Boolean(user.hasCompletedOnboarding),
      },
      profile: userProfile || null,
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

    // Retrieve onboarding profile from MongoDB
    const userProfile = await Profile.findOne({ userId: user._id });
    const appToken = jwt.sign(
      { userId: user._id, email: user.email },
      process.env.JWT_SECRET || "your_jwt_secret",
      { expiresIn: "7d" }
    );

    return res.json({
      token: appToken,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        hasCompletedOnboarding: Boolean(user.hasCompletedOnboarding),
      },
      profile: userProfile || null,
    });
  } catch (error) {
    console.error("Google auth server error:", error);
    return res.status(500).json({ message: "Internal server error during Google login" });
  }
});

// POST /api/auth/profile - Save onboarding choices to MongoDB
router.post("/profile", authenticateToken, async (req, res) => {
  try {
    const { role, level } = req.body;

    const profile = await Profile.findOneAndUpdate(
      { userId: req.user.userId },
      { role, level },
      { returnDocument: "after", upsert: true }
    );

    res.json({ profile });
  } catch (err) {
    console.error("Error saving profile:", err);
    res.status(500).json({ error: "Failed to save profile onboarding info" });
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

    const baseUrl = "http://localhost:8080";
    const resetUrl = `${baseUrl}/reset-password?token=${resetToken}&id=${user._id}`;

    await transporter.sendMail({
      from: `"Intervue You" <${process.env.EMAIL_USER}>`,
      to: user.email,
      subject: "Reset your Intervue You password",
      text: `You requested a password reset for Intervue You. Please use the following link to reset your password (valid for 15 minutes):\n\n${resetUrl}\n\nIf you did not request this, please ignore this email.`,
      html: `
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 560px; margin: 0 auto; padding: 24px; border: 1px solid #e4e4e7; border-radius: 16px; background-color: #ffffff; color: #18181b;">
          <h2 style="font-size: 20px; font-weight: 600; margin-top: 0; color: #09090b;">Reset your password</h2>
          <p style="font-size: 15px; line-height: 1.5; color: #52525b; margin-bottom: 24px;">
            We received a request to reset the password for your Intervue You account. Click the button below to choose a new password:
          </p>
          <div style="margin: 28px 0;">
            <a href="${resetUrl}" target="_blank" style="background-color: #2563eb; color: #ffffff; padding: 12px 24px; text-decoration: none; border-radius: 9999px; font-weight: 500; font-size: 14px; display: inline-block;">Reset Password</a>
          </div>
          <p style="font-size: 13px; color: #71717a; line-height: 1.5;">
            This link is valid for 15 minutes. If you did not request a password reset, you can safely ignore this email.
          </p>
          <hr style="border: none; border-top: 1px solid #f4f4f5; margin: 24px 0;" />
          <p style="font-size: 12px; color: #a1a1aa; margin: 0;">Intervue You — AI Mock Interview Platform</p>
        </div>
      `,
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

// POST /api/auth/complete-onboarding
router.post("/complete-onboarding", async (req, res) => {
  try {
    const userId = extractUserId(req);

    if (!userId) {
      return res.status(401).json({ error: "Unauthorized: Missing or invalid authentication token" });
    }

    const { role, level, cv } = req.body || {};

    const updatedUser = await User.findByIdAndUpdate(
      userId,
      {
        hasCompletedOnboarding: true,
        ...(role && { role }),
        ...(level && { level }),
        ...(cv && { cv }),
      },
      { returnDocument: "after" }
    );

    if (!updatedUser) {
      return res.status(404).json({ error: "User profile not found in database" });
    }

    console.log(`✅ Onboarding completed for user: ${updatedUser.email || userId}`);
    return res.json({ success: true, user: updatedUser });
  } catch (err) {
    console.error("❌ Onboarding completion backend error:", err);
    return res.status(500).json({
      error: "Failed to update onboarding status",
      details: err.message,
    });
  }
});

module.exports = router;