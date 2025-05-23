#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

const float EPS = 1e-5f;

typedef struct {
  int dim; // transformer dim
  int hidden_dim; // for ffn layers
  int n_layers; // number of layers
  int q_heads; // num of query heads
  int kv_heads; // num of key/value heads (for GQA and MQA whose kv_heads <
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
  float* xb;
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

void memory_map_weights(TransformerWeights* w, Config* config, float* weight_ptr, int tie_embedding) {
  int dim = config->dim;
  int head_size = dim / config->q_heads;
  size_t n_layers = config->n_layers;

  w->token_embedding_table = weight_ptr;
  weight_ptr += config->vocab_size * dim; // move to next weight

  w->rms_attn_weight = weight_ptr;
  weight_ptr += n_layers * dim;

  // attn
  w->wq = weight_ptr;
  weight_ptr += n_layers * dim * config->q_heads * head_size;
  w->wk = weight_ptr;
  weight_ptr += n_layers * dim * config->kv_heads * head_size;
  w->wv = weight_ptr;
  weight_ptr += n_layers * dim * config->kv_heads * head_size;
  w->wo = weight_ptr;
  weight_ptr += n_layers * dim * dim;

  // ffn
  w->rms_ffn_weight = weight_ptr;
  weight_ptr += n_layers * dim;
  w->gate = weight_ptr;
  weight_ptr += n_layers * dim * config->hidden_dim;
  w->down = weight_ptr;
  weight_ptr += n_layers * dim * config->hidden_dim;
  w->up = weight_ptr;
  weight_ptr += n_layers * dim * config->hidden_dim;

  w->rms_final_weight = weight_ptr;
  weight_ptr += dim;
  // skip what used to be freq_cls_real (for RoPE)
  weight_ptr += config->seq_len * head_size / 2;
  weight_ptr += config->seq_len * head_size / 2;
  w->wcls = tie_embedding ? w->token_embedding_table : weight_ptr;
}

void read_checkpoint(const char* checkpoint_path, Config* config, TransformerWeights* transformer_weight, ssize_t* file_size) {
  FILE* file = fopen(checkpoint_path, "rb");
  if (file == NULL) {
    printf("Couldn't open file %s\n", checkpoint_path);
    exit(EXIT_FAILURE);
  }
  printf("lxylog dim: %d\n", config->dim);
  size_t config_items_read = fread(config, sizeof(Config), 1, file);
  #if DEBUG > 0
  printf("config read size %ld\n", config_items_read);
  #endif
  if (config_items_read != 1) {
    printf("Failed to read config from file\n");
    fclose(file);
    exit(EXIT_FAILURE);
  }
  #if DEBUG > 0
  printf("Config:\n");
  printf("  dim: %d\n", config->dim);
  printf("  hidden_dim: %d\n", config->hidden_dim);
  printf("  n_layers: %d\n", config->n_layers);
  printf("  n_heads: %d\n", config->q_heads);
  printf("  n_kv_heads: %d\n", config->kv_heads);
  printf("  vocab_size: %d\n", config->vocab_size);
  printf("  seq_len: %d\n", config->seq_len);
  #endif

  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int tie_embedding = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  fseek(file, 0, SEEK_END); // move file pointer to end of file
  *file_size = ftell(file); // get the file size, in bytes
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

  float* weight_ptr = (float*)((char*)mapped_data + sizeof(Config));
  memory_map_weights(transformer_weight, config, weight_ptr, tie_embedding);
}

void malloc_run_state(const Config* config, RunState* state) {
  size_t vec_size = (size_t)(config->seq_len * config->dim);
  state->x = calloc(vec_size, sizeof(float));
  state->xb = calloc(vec_size, sizeof(float));
  if (!state->x || !state->xb) {
    fprintf(stderr, "Failed to allocate memory for RunState\n");
    exit(EXIT_FAILURE);
  }
}

void build_transformer(Transformer* transformer, const char* checkpoint_path) {
  ssize_t file_size;
  read_checkpoint(checkpoint_path, &transformer->config, &transformer->weights, &file_size);
  malloc_run_state(&transformer->config, &transformer->state);
}

void rmsnorm(float* out, float* x, const float* weight, const int B, const int T, const int C) {
  for (int b = 0; b < B; ++b) {
    // move ptr of x and out
    x += b * T * C;
    out += b * T * C;
    for (int t = 0; t < T; ++t) {
      float* inp_t = x + t * C;
      float* out_t = out + t * C;
      float ss = 0.f; // sum of squares
      for (int i = 0; i < C; ++i) { ss += inp_t[i] * inp_t[i]; }
      ss /= C;
      ss = 1.f / sqrtf(ss + EPS); // (rsqrt)
      for (int i = 0; i < C; ++i) { out_t[i] = inp_t[i] * ss * weight[i]; }
    }
  }
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
float* llm_forward(Transformer* transformer, int token, int pos) {
  const Config* config = &transformer->config;
  const TransformerWeights* weights = &transformer->weights;
  const RunState* state = &transformer->state;

  float* x = state->x;
  float* xb = state->xb;
  int dim = config->dim;

  // token lookup table
  float* embed_token = weights->token_embedding_table + (token * dim);
  memcpy(x, embed_token, dim * sizeof(float));
  printf("embedding finish, first element is %f\n", x[0]);

  for (int i = 0; i < config->n_layers; ++i) {
    rmsnorm(xb, x, weights->rms_attn_weight, 1, 1, dim);
  }
  return x;
}

// Question 5: Implement RoPE in forward function
// (This will be part of the forward function above)

// Question 6: Implement sample
int sample(Sampler* sampler, float* logits) {
  // Your implementation here
  return 0;
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
  Transformer transformer;
  Tokenizer tokenizer;
  Sampler sampler;

  /* Expect exactly one positional argument: the checkpoint file to load. */
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <checkpoint_path>\n", argv[0]);
    return EXIT_FAILURE;
  }
  const char* checkpoint_path = argv[1];
#if DEBUG > 0
  printf("loading checkpoint path %s\n", checkpoint_path);
#endif
  build_transformer(&transformer, checkpoint_path);

  float* logits = llm_forward(&transformer, 10, 0);
  // Clean up and free memory

  return EXIT_SUCCESS;
}
