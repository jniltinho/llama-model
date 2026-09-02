// llama-model lists the GGUF models Ollama has pulled and switches the one
// llama-server runs, keeping the OpenCode and Hermes configs in sync.
package main

import "llama-model/cmd"

func main() {
	cmd.Execute()
}
