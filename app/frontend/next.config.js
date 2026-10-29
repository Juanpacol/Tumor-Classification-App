/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack(config) {
    // Cornerstone3D Tools imports a WebAssembly module for polygon
    // segmentation, which the portal does not use and webpack cannot bundle.
    config.resolve.alias["@icr/polyseg-wasm"] = false;
    return config;
  },
};

module.exports = nextConfig;
