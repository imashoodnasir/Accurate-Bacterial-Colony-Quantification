
import random, copy

def pso(objective, space: dict, iters=10, swarm=10):
    def sample_one():
        s = {}
        for k,v in space.items():
            if isinstance(v, (list, tuple)):
                s[k] = random.choice(v)
            else:
                s[k] = v()
        return s

    gbest, gscore = None, float("inf")
    particles = [sample_one() for _ in range(swarm)]
    pbest = [copy.deepcopy(p) for p in particles]
    pscore= [float("inf")] * swarm

    for it in range(iters):
        for i,p in enumerate(particles):
            score = objective(p)
            if score < pscore[i]:
                pbest[i], pscore[i] = copy.deepcopy(p), score
            if score < gscore:
                gbest, gscore = copy.deepcopy(p), score

        for i,p in enumerate(particles):
            for k in p:
                if random.random() < 0.5: p[k] = pbest[i][k]
                if random.random() < 0.3: p[k] = gbest[k]
                if random.random() < 0.2:
                    p[k] = sample_one()[k]
        print(f"Iter {it+1}/{iters}: best={gscore:.4f}, gbest={gbest}")
    return gbest, gscore
