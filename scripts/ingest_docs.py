"""
Document ingestion script for the RAG Evaluation System.
Ingests one or more files into the ChromaDB vector store via the API.

Usage:
    python scripts/ingest_docs.py path/to/doc1.pdf path/to/doc2.txt
    """

    import sys
    import httpx
    from pathlib import Path

    BASE_URL = "http://localhost:8000"


    def ingest_files(file_paths: list[str]) -> None:
        """Send file paths to the ingest API endpoint."""
            valid_paths = []
                for fp in file_paths:
                        path = Path(fp)
                                if not path.exists():
                                            print(f"[WARNING] File not found: {fp}")
                                                        continue
                                                                if path.suffix.lower() not in (".pdf", ".txt"):
                                                                            print(f"[WARNING] Unsupported file type: {fp} (only .pdf and .txt supported)")
                                                                                        continue
                                                                                                valid_paths.append(str(path.resolve()))

                                                                                                    if not valid_paths:
                                                                                                            print("No valid files to ingest.")
                                                                                                                    return
                                                                                                                    
                                                                                                                        print(f"Ingesting {len(valid_paths)} file(s)...")
                                                                                                                            response = httpx.post(
                                                                                                                                    f"{BASE_URL}/api/ingest",
                                                                                                                                            json={"file_paths": valid_paths},
                                                                                                                                                    timeout=60.0,
                                                                                                                                                        )
                                                                                                                                                        
                                                                                                                                                            if response.status_code == 200:
                                                                                                                                                                    data = response.json()
                                                                                                                                                                            print(f"Success: {data}")
                                                                                                                                                                                else:
                                                                                                                                                                                        print(f"Error {response.status_code}: {response.text}")
                                                                                                                                                                                        
                                                                                                                                                                                        
                                                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                                                            if len(sys.argv) < 2:
                                                                                                                                                                                                    print("Usage: python scripts/ingest_docs.py <file1> [file2] ...")
                                                                                                                                                                                                            sys.exit(1)
                                                                                                                                                                                                                ingest_files(sys.argv[1:])
                                                                                                                                                                                                                
