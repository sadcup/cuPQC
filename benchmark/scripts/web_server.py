#!/usr/bin/env python3
"""
Simple HTTP server for cuPQC Benchmark Report
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8080
DIRECTORY = "/workspace/benchmark/results"


class BenchmarkHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Add CORS headers if needed
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()


def main():
    # Change to results directory
    os.chdir(DIRECTORY)
    
    print(f"Starting cuPQC Benchmark Report Server...")
    print(f"Serving at http://localhost:{PORT}/")
    print(f"Directory: {DIRECTORY}")
    print(f"\nPress Ctrl+C to stop")
    
    with socketserver.TCPServer(("", PORT), BenchmarkHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            return 0


if __name__ == "__main__":
    exit(main())
