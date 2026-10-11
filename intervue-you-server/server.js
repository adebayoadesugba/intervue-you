require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const authRoutes = require("./src/routes/auth");
const sessionRoutes = require("./src/routes/sessions");
const app = express();

// Middleware
app.use(express.json());

const allowedOrigins = [
  process.env.DEV_ENV || "http://localhost:8080",
  process.env.PROD_ENV || "https://intervue-you.netlify.app",
  "https://intervue-you.netlify.app",
  process.env.CLIENT_URL, // Optional: add env variable for flexibility
].filter(Boolean);

app.use(
  cors({
    origin: function (origin, callback) {
      // Allow requests with no origin (like mobile apps or Postman)
      if (!origin) return callback(null, true);
      
      if (allowedOrigins.includes(origin)) {
        return callback(null, true);
      } else {
        return callback(new Error("Not allowed by CORS"));
      }
    },
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);
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