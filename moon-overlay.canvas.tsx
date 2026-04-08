export default function MoonOverlay() {
  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background:
          "radial-gradient(ellipse at 50% 30%, #0a1628 0%, #020810 100%)",
        position: "relative",
        overflow: "hidden",
        fontFamily: "'Georgia', 'Times New Roman', serif",
      }}
    >
      {/* Audio — autoplay the song */}
      <audio
        src="http://localhost:8111/Christina%20Perri%20-%20A%20Thousand%20Years.mp3"
        autoPlay
        loop
        style={{ display: "none" }}
      />

      {/* Stars */}
      {Array.from({ length: 60 }).map((_, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            width: i % 5 === 0 ? 3 : 1.5,
            height: i % 5 === 0 ? 3 : 1.5,
            borderRadius: "50%",
            background: "white",
            top: `${(i * 17.3) % 100}%`,
            left: `${(i * 23.7) % 100}%`,
            opacity: 0.3 + (i % 4) * 0.2,
            animation: `twinkle ${2 + (i % 3)}s ease-in-out infinite`,
            animationDelay: `${(i * 0.3) % 4}s`,
          }}
        />
      ))}

      {/* Moon glow */}
      <div
        style={{
          position: "absolute",
          top: "8%",
          right: "15%",
          width: 120,
          height: 120,
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(255,248,220,0.9) 0%, rgba(255,248,220,0.3) 40%, transparent 70%)",
          boxShadow:
            "0 0 80px 40px rgba(255,248,220,0.15), 0 0 160px 80px rgba(255,248,220,0.05)",
        }}
      />

      {/* Ocean shimmer at bottom */}
      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          height: "25%",
          background:
            "linear-gradient(to top, rgba(10,40,80,0.6) 0%, rgba(10,40,80,0.2) 50%, transparent 100%)",
        }}
      />
      <div
        style={{
          position: "absolute",
          bottom: "12%",
          left: "50%",
          transform: "translateX(-50%)",
          width: "80%",
          height: 1,
          background:
            "linear-gradient(to right, transparent, rgba(180,210,255,0.3), transparent)",
          animation: "shimmer 4s ease-in-out infinite",
        }}
      />

      {/* Main text */}
      <div
        style={{
          textAlign: "center",
          zIndex: 10,
          padding: "0 40px",
          animation: "fadeIn 3s ease-in-out",
        }}
      >
        <p
          style={{
            fontSize: 28,
            color: "rgba(200, 220, 255, 0.85)",
            letterSpacing: 6,
            textTransform: "uppercase",
            marginBottom: 40,
            fontWeight: 300,
          }}
        >
          The moon sings
        </p>

        <p
          style={{
            fontSize: 22,
            color: "rgba(180, 200, 235, 0.65)",
            letterSpacing: 3,
            marginBottom: 60,
            fontWeight: 300,
          }}
        >
          and the ocean is pulled
        </p>

        <div
          style={{
            width: 120,
            height: 1,
            background:
              "linear-gradient(to right, transparent, rgba(180,210,255,0.5), transparent)",
            margin: "0 auto 60px",
          }}
        />

        <p
          style={{
            fontSize: 18,
            color: "rgba(160, 185, 220, 0.6)",
            letterSpacing: 2,
            lineHeight: 2.2,
            maxWidth: 500,
            margin: "0 auto",
            fontWeight: 300,
            fontStyle: "italic",
          }}
        >
          So here's a song from the ocean
          <br />
          for your moon
        </p>
      </div>

      {/* Floating particles */}
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={`p-${i}`}
          style={{
            position: "absolute",
            width: 2,
            height: 2,
            borderRadius: "50%",
            background: "rgba(180,210,255,0.4)",
            bottom: `${15 + ((i * 7) % 20)}%`,
            left: `${10 + ((i * 13) % 80)}%`,
            animation: `float ${6 + (i % 4) * 2}s ease-in-out infinite`,
            animationDelay: `${i * 0.8}s`,
          }}
        />
      ))}

      <style>{`
        @keyframes twinkle {
          0%, 100% { opacity: 0.2; }
          50% { opacity: 0.8; }
        }
        @keyframes fadeIn {
          0% { opacity: 0; transform: translateY(20px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0) translateX(0); opacity: 0.2; }
          50% { transform: translateY(-40px) translateX(10px); opacity: 0.6; }
        }
        @keyframes shimmer {
          0%, 100% { opacity: 0.3; width: 60%; }
          50% { opacity: 0.6; width: 90%; }
        }
      `}</style>
    </div>
  );
}
