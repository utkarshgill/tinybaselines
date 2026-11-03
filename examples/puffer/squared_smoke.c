#include <assert.h>

#include "ocean/squared_continuous/squared_continuous.h"

int main(void) {
    obs_t observations[OBS_SIZE] = {0};
    float actions[NUM_ATNS] = {0}, reward = 0, terminal = 0;
    DictItem size = {.key = "size", .value = 11};
    Dict options = {.items = &size, .size = 1, .cap = 1};
    Squared env = {.rng = 1};
    env.agents[0] = (Agent){.observations = observations, .actions = actions,
                            .rewards = &reward, .terminals = &terminal};
    puf_init(&env, &options);
    puf_reset(&env);

    const int center = OBS_SIZE / 2;
    assert(env.num_agents == 1 && observations[center] == AGENT && !IsWindowReady());
    memset(observations, EMPTY, sizeof(observations));
    observations[center] = AGENT;
    observations[center + 1] = TARGET;

    actions[1] = 0.2f;
    puf_step(&env);
    assert(reward == 0 && terminal == 0 && observations[center] == AGENT);
    actions[1] = 0.3f;
    puf_step(&env);
    assert(reward == 1 && terminal == 1 && env.tick == 0 && observations[center] == AGENT);

    actions[1] = 0;
    for (int i = 0; i < 3 * env.size; i++) {
        puf_step(&env);
        assert(reward == 0 && terminal == 0);
    }
    puf_step(&env);
    assert(reward == -1 && terminal == 1 && env.tick == 0 && observations[center] == AGENT);
    assert(!IsWindowReady());
    puf_close(&env);
    puts("Squared Continuous: movement threshold, goal, timeout, and same-step reset passed.");
}
