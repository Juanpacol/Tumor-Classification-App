/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ];
  },
  webpack(config) {
    // Cornerstone3D Tools imports a WebAssembly module for polygon
    // segmentation, which the portal does not use and webpack cannot bundle.
    config.resolve.alias["@icr/polyseg-wasm"] = false;
    return config;
  },
};

module.exports = nextConfig;
