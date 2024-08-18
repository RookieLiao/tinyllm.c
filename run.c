#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Structures

typedef struct {
  int dim;        // transformer dim
  int hidden_dim; // for ffn layers
  int n_layers;   // number of layers
  int n_heads;    // num of query heads
  int n_kv_heads; // num of key/value heads (for GQA and MQA whose kv_heads <
                  // query_heads)
  int vocab_size; // vocabulary size
  int seq_len;    // max sequence length
} Config;

typedef struct {
  // token embedding weights
  float *token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  float *rms_attn_weight; // (layer, dim)
  float *rms_ffn_weight;  // (layer, dim)
  // weights for attention, note dim == n_heads * head_size
  float *wq; // (layer, dim, n_heads * head_size)
  float *wk; // (layer, dim, kv_heads * head_size)
  float *wv; // (layer, dim, kv_heads * head_size)
  float *wo; // (layer, n_heads * head_size, dim)
  // weights for FFN
  float *up;   // (layer, hidden_dim, dim)
  float *gate; // (layer, dim, 4*dim)
  float *down; // (layer, 4*dim, dim)
  // final rmsnorm
  float *rms_final_weight; // (dim,)
  // (Optional): classifier weights for the logits, on the last layer
  float *wcls; // (dim, vocab_size)
} TransformerWeights;

typedef struct {
    // Add necessary fields for run state
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
} Transformer;

typedef struct {
    // Add necessary fields for tokenizer
} Tokenizer;

typedef struct {
    // Add necessary fields for sampler
} Sampler;

// Function declarations

// Question 1: Implement rmsnorm
void rmsnorm(float* o, float* x, float* weight, int size) {
    // Your implementation here
}

// Question 2: Implement softmax
void softmax(float* x, int size) {
    // Your implementation here
}

// Question 3: Implement matmul
void matmul(float* xout, float* x, float* w, int n, int d) {
    // Your implementation here
}

// Question 4: Implement forward (partial)
float* forward(Transformer* transformer, int token, int pos) {
    // Your implementation here
}

// Question 5: Implement RoPE in forward function
// (This will be part of the forward function above)

// Question 6: Implement sample
int sample(Sampler* sampler, float* logits) {
    // Your implementation here
}

// Question 7: Implement encode
void encode(Tokenizer* t, char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens) {
    // Your implementation here
}

// Question 8: Implement decode
char* decode(Tokenizer* t, int prev_token, int token) {
    // Your implementation here
}

// Question 9: Implement generate
void generate(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* prompt, int steps) {
    // Your implementation here
}

// Question 10: Implement chat
void chat(Transformer* transformer, Tokenizer* tokenizer, Sampler* sampler, char* cli_user_prompt, char* cli_system_prompt, int steps) {
    // Your implementation here
}

// Main function
int main(int argc, char* argv[]) {
    // Initialize structures
    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;

    // Parse command line arguments and set up configuration

    // Call generate or chat based on mode

    // Clean up and free memory

    return 0;
}
