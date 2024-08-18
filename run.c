#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
// Structures

typedef struct {
  int dim; // transformer dim
  int hidden_dim; // for ffn layers
  int n_layers; // number of layers
  int n_heads; // num of query heads
  int n_kv_heads; // num of key/value heads (for GQA and MQA whose kv_heads <
                  // query_heads)
  int vocab_size; // vocabulary size
  int seq_len; // max sequence length
} Config;

typedef struct {
  // token embedding weights
  float* token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  float* rms_attn_weight; // (layer, dim)
  float* rms_ffn_weight; // (layer, dim)
  // weights for attention, note dim == n_heads * head_size
  float* wq; // (layer, dim, n_heads * head_size)
  float* wk; // (layer, dim, kv_heads * head_size)
  float* wv; // (layer, dim, kv_heads * head_size)
  float* wo; // (layer, n_heads * head_size, dim)
  // weights for FFN
  float* up; // (layer, hidden_dim, dim)
  float* gate; // (layer, dim, 4*dim)
  float* down; // (layer, 4*dim, dim)
  // final rmsnorm
  float* rms_final_weight; // (dim,)
  // (Optional): classifier weights for the logits, on the last layer
  float* wcls; // (dim, vocab_size)
} TransformerWeights;

typedef struct {
  float* x; // activation
  float* xr;
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

void memory_map_weights(
    TransformerWeights* w,
    Config* cfg,
    float* w_ptr,
    int tie_embedding) {
  int dim = cfg->dim;
  int head_size = dim / cfg->n_heads;
  size_t n_layers = cfg->n_layers;

  w->token_embedding_table = w_ptr;
  w_ptr += cfg->vocab_size * dim;

  w->rms_attn_weight = w_ptr;
  w_ptr += n_layers * dim;

  // attn
  w->wq = w_ptr;
  w_ptr += n_layers * dim * dim;
  w->wk = w_ptr;
  w_ptr += n_layers * dim * cfg->n_kv_heads * head_size;
  w->wv = w_ptr;
  w_ptr += n_layers * dim * cfg->n_kv_heads * head_size;
  w->wo = w_ptr;
  w_ptr += n_layers * dim * dim;

  // ffn
  w->rms_ffn_weight = w_ptr;
  w_ptr += n_layers * dim;
  w->gate = w_ptr;
  w_ptr += n_layers * dim * cfg->hidden_dim;
  w->down = w_ptr;
  w_ptr += n_layers * dim * cfg->hidden_dim;
  w->up = w_ptr;
  w_ptr += n_layers * dim * cfg->hidden_dim;

  w->rms_final_weight = w_ptr;
  w_ptr += dim;
  w_ptr += cfg->seq_len * head_size /
      2; // skip what used to be freq_cls_real (for RoPE)
  w_ptr += cfg->seq_len * head_size /
      2; // skip what used to be freq_cls_imag(for RoPE)
  w->wcls = tie_embedding ? w->token_embedding_table : w_ptr;
  printf("w_cls weight 4: %f\n", w->wcls[3]);
}

void read_checkpoint(
    const char* checkpoint_path,
    TransformerWeights* transformer_weight,
    ssize_t* file_size) {
  printf("lxylog\n");
  FILE* file = fopen(checkpoint_path, "rb");
  if (file == NULL) {
    printf("Couldn't open file %s\n", checkpoint_path);
    exit(EXIT_FAILURE);
  }
  printf("lxylog for opening\n");
  Config* config = malloc(sizeof(Config));
  size_t read = fread(config, sizeof(Config), 1, file);
  printf("lxylog for read size %ld\n", read);
  if (read != 1) {
    printf("Failed to read config from file\n");
    fclose(file);
    exit(EXIT_FAILURE);
  }
  // Print all Config attributes
  printf("Config:\n");
  printf("  dim: %d\n", config->dim);
  printf("  hidden_dim: %d\n", config->hidden_dim);
  printf("  n_layers: %d\n", config->n_layers);
  printf("  n_heads: %d\n", config->n_heads);
  printf("  n_kv_heads: %d\n", config->n_kv_heads);
  printf("  vocab_size: %d\n", config->vocab_size);
  printf("  seq_len: %d\n", config->seq_len);

  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int tie_embedding = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes

  printf("total size %ld\n", *file_size);
  fclose(file);

  int fd = open(checkpoint_path, O_RDONLY);
  if (fd == -1) {
    printf("open failed for %s\n", checkpoint_path);
    exit(EXIT_FAILURE);
  }

  void* mapped_data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped_data == MAP_FAILED) {
    printf("mmap failed!");
    exit(EXIT_FAILURE);
  }

  // float* weight = (float*)((char*)mapped_data + sizeof(Config));
  float* weight = (float*)((char*)mapped_data + sizeof(Config));
  memory_map_weights(transformer_weight, config, weight, tie_embedding);
}

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
void encode(
    Tokenizer* t,
    char* text,
    int8_t bos,
    int8_t eos,
    int* tokens,
    int* n_tokens) {
  // Your implementation here
}

// Question 8: Implement decode
char* decode(Tokenizer* t, int prev_token, int token) {
  // Your implementation here
}

// Question 9: Implement generate
void generate(
    Transformer* transformer,
    Tokenizer* tokenizer,
    Sampler* sampler,
    char* prompt,
    int steps) {
  // Your implementation here
}

// Question 10: Implement chat
void chat(
    Transformer* transformer,
    Tokenizer* tokenizer,
    Sampler* sampler,
    char* cli_user_prompt,
    char* cli_system_prompt,
    int steps) {
  // Your implementation here
}

// Main function
int main(int argc, char* argv[]) {
  // Initialize structures
  Transformer transformer;
  Tokenizer tokenizer;
  Sampler sampler;

  // Parse command line arguments and set up configuration
  char* checkpoint_path = NULL;
  // ./run stories42M.bin -t 0.8 -n 256 -i "One day, Lily met a Shoggoth"
  if (argc > 1) {
    checkpoint_path = argv[1];
  }
  printf("checkpoint path %s\n", checkpoint_path);
  ssize_t file_size;
  TransformerWeights transformer_weight;
  read_checkpoint(checkpoint_path, &transformer_weight, &file_size);

  // Call generate or chat based on mode

  // Clean up and free memory

  return EXIT_SUCCESS;
}
