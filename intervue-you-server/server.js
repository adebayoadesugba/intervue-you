require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const authRoutes = require("./src/routes/auth");
const sessionRoutes = require("./src/routes/sessions");
const app = express();

// Middleware
app.use(express.json());
app.use(cors({
  origin: "http://localhost:8080", // Your Vite React frontend URL
  credentials: true
}));

// Routes
app.use("/api/auth", authRoutes);
app.use("/api/sessions", sessionRoutes);

// Database Connection
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("✅ MongoDB Connected Successfully"))
  .catch((err) => console.error("❌ MongoDB Connection Error:", err));

// Base route
app.get("/", (req, res) => {
  res.send("Intervue-You Auth API is running!");
});

// Basic Health Check Route
app.get("/api/health", (req, res) => {
  res.json({ status: "Server is running", database: mongoose.connection.readyState === 1 ? "Connected" : "Disconnected" });
});



// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Auth & User Server running on http://localhost:${PORT}`);
});