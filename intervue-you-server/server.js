require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

// Routes
const authRoutes = require("./src/routes/auth");
const sessionRoutes = require("./src/routes/sessions");
const interviewRouter = require("./src/routes/interview"); // Fixed path to ./src/routes/interview

const app = express();

// Allowed Origins for CORS
const allowedOrigins = [
  process.env.DEV_ENV || "http://localhost:8080",
  process.env.PROD_ENV || "https://intervue-you.netlify.app",
  "http://localhost:5173",
  "https://intervue-you.netlify.app",
  process.env.CLIENT_URL,
].filter(Boolean);

// CORS Middleware Configuration
app.use(
  cors({
    origin: function (origin, callback) {
      // Allow requests with no origin (like mobile apps, Postman, or curl)
      if (!origin || allowedOrigins.includes(origin)) {
        return callback(null, true);
      }
      return callback(null, false);
    },
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// Body Parser Middleware (MUST BE PLACED BEFORE ROUTES)
app.use(express.json());

// Mount API Routes
app.use("/api/auth", authRoutes);
app.use("/api/sessions", sessionRoutes);
app.use("/api/interview", interviewRouter);

// Database Connection
mongoose
  .connect(process.env.MONGO_URI, {
    family: 4,                       // Force IPv4 (prevents Windows IPv6 stall/timeout)
    maxPoolSize: 10,
    minPoolSize: 1,
    maxIdleTimeMS: 30000,
    serverSelectionTimeoutMS: 30000, // Give Atlas M0 replica-set elections 30s to complete
    connectTimeoutMS: 20000,
    socketTimeoutMS: 45000,
    retryWrites: true,
    retryReads: true,
  })
  .then(() => {
    console.log("✅ MongoDB Connected Successfully");
    app.listen(PORT, () => {
      console.log(`🚀 Server running on http://localhost:${PORT}`);
    });
  })
  .catch((err) => {
    console.error("❌ MongoDB Connection Error:", err);
    process.exit(1);
  });

// Base Route
app.get("/", (req, res) => {
  res.send("Intervue-You Auth API is running!");
});

// Basic Health Check Route
app.get("/api/health", (req, res) => {
  res.json({
    status: "Server is running",
    database: mongoose.connection.readyState === 1 ? "Connected" : "Disconnected",
  });
});

// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});