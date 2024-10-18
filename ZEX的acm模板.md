# ZEX的acm模板

[TOC]



## 图论

### 表达式树

```c++
const int maxn = 1000;
int lch[maxn], rch[maxn];
char op[maxn];
int nc = 0;
int build_tree(char *s, int x,int y) {
    int i, c1 = -1, c2 = -1, p = 0;
    int u;
    if(y - x == 1) {//一个字符，建立单独节点
        u = ++ nc;
        lch[u] = rch[u] = 0;
        op[u] = s[x];
        return u;
    }
    for(i = x; i < y; i ++) {
        switch (s[i])
        {
        case '(': p ++; break;
        case ')': p --; break;
        case '+': case '-': if(!p) c1 = i; break;
        case '*': case '/': if(!p) c2 = i; break;
        }
    }
    if(c1 < 0) c1 = c2;//找不到括号外的加减号就用乘除号
    if(c1 < 0) return build_tree(s, x + 1, y - 1);//整个表达式被括号括起来
    u = ++ nc;
    lch[u] = build_tree(s, x, c1);
    rch[u] = build_tree(s, c1 + 1, y);
    op[u] = s[c1];
    return u;
}

```

### kruskal算法

```cpp
int cmp(const int i, const int j) {return w[i] < w[j];}
int find(int x) {return p[x] == x ? x : p[x] = find(p[x]);}
int kruskal() {
    int ans = 0;
    for(int i = 0; i < n; i ++) p[i] = i;
    for(int i = 0; i < m; i ++) r[i] = i;
    sort(r, r + m, cmp);
    for(int i = 0; i < m; i ++) {
        int e = r[i]; int x = find(u[e]); int y = find(v[e]);
        if(x != y) {ans += w[e]; p[x] = y;}
    }
    return ans;
}
```

### 接邻链表存图

```cpp
int d[N];
int n, m;
int first[N];
int u[N], v[N], w[N], next[N];
void read_graph() {
    scanf("%d%d", &n, &m);
    for(int i = 0; i < n; i ++) first[i] = -1;
    for(int e = 0; e < m; e ++) {
        scanf("%d%d%d", &u[e], &v[e], &w[e]);
        next[e] = first[u[e]];
        first[u[e]] = e;
    }
}
```

### Dijkstra

```cpp
const int N = 2e5 + 10;
const int inf = 0x3f3f3f3f;
using namespace std;
struct Edge {
    int from, to, dist;
    Edge(int u, int v, int d):from(u), to(v), dist(d) {}
};

struct HeapNode {
    int d, u;
    bool operator < (const HeapNode& rhs) const {
        return d >rhs.d;
    }
};

class Dijkstra {
private:
    int n, m;
    vector<Edge> edges;
    vector<int> G[N];
    bool done[N];//是否已永久标号
    int d[N];//s到各个点的距离
    int p[N];//最短路中的上一条弧

public:
    void init(int n) {
        this->n = n;
        for(int i = 0; i < n; i ++) G[i].clear();
        edges.clear();
    }

    void AddEdge(int from, int to, int dist) {
        edges.push_back(Edge(from, to, dist));
        m = edges.size();
        G[from].push_back(m - 1);
    }
    
    void dijkstra(int s) {
        priority_queue<HeapNode> Q;
        for(int i = 0; i < n; i ++) d[i] = inf;
        d[s] = 0;
        memset(done, 0, sizeof done);
        Q.push((HeapNode){0, s});
        while(!Q.empty()) {
            HeapNode x = Q.top(); Q.pop();
            int u = x.u;
            if(done[u]) continue;
            done[u] = true;
            for(int i = 0; i < G[u].size(); i ++) {
                Edge& e = edges[G[u][i]];
                if(d[e.to] > d[u] + e.dist) {
                    d[e.to] = d[u] + e.dist;
                    p[e.to] = G[u][i];
                    Q.push((HeapNode){d[e.to], e.to});
                }
            }
        }
    }
};

```

#### 较短版本

```c++
void dijkstra() {
    priority_queue<pi, vector<pi>, greater<pi>> que;
    que.push({0, n});
    while(!que.empty()) {
        auto fro = que.top(); que.pop();
        int pos = fro.se, dis = fro.fi;
        if(vis[pos]) continue;
        vis[pos] = 1;
        d[pos] = dis;
        for(auto v : e[pos]) {
            que.push({dis + deg[v], v});
            deg[v] --;
        }
    }
}
```



### Bellmen-Ford

```cpp
struct Edge{
	int from, to, dist;
	Edge(int u, int v, int d) : from(u), to(v), dist(d) {}
};

int cnt[maxn], d[maxn], p[maxn], n, m;
bool inq[maxn];
vector<int> G[maxn];
vector<Edge> edges; 

void AddEdge(int x, int y, int z) {
	edges.push_back(Edge(x, y, z));
	int t = (int)edges.size();
	G[x].push_back(t - 1);
}
bool Bellman_ford(int s) {
    queue<int> Q;
    memset(inq, 0, sizeof inq);
    memset(cnt, 0, sizeof cnt);
    for(int i = 0; i < n; i ++) d[i] = inf;
    d[s] = 0;
    inq[s] = true;
    Q.push(s);

    while(!Q.empty()) {
        int u = Q.front();Q.pop();
        inq[u] = false;
        for(int i = 0; i < G[u].size(); i ++) {
            Edge& e = edges[G[u][i]];
            if(d[u] < inf && d[e.to] > d[u] + e.dist) {
                d[e.to] = d[u] + e.dist;
                p[e.to] = G[u][i];
                if(!inq[e.to]) {
                    Q.push(e.to);inq[e.to] = true;
                    if(++ cnt[e.to] > n) return false;
                }
            }
        }
    }
    return true;
}
```

### floyd

```cpp
void folyd() {
    for(int k = 0; k < n; k ++)
        for(int i = 0; i < n; i ++) 
            for(int j = 0; j < n; j ++) 
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]); 
}

```

### spfa

```c++
void spfa(int s) {
    memset(dist, 127, sizeof dist);
    queue<A> que;
    que.push(A{s, 0});
    b[s][0] = true;
    dist[s][0] = 0;
    while(!que.empty()) {
        auto f = que.front();
        que.pop();
        b[f[0]][f[1]] = false;
        for(auto v : e[f[0]]) {
            if(dist[v[0]][f[1]] > dist[f[0]][f[1]] + v[1]) {
                dist[v[0]][f[1]] = dist[f[0]][f[1]] + v[1];
                // printf("from: %d %d to: %d %d dis: %d\n", f[0], f[1], v[0], f[1], dist[v[0]][f[1]]);
                if(!b[v[0]][f[1]]) {
                    que.push(A{v[0], f[1]});
                    b[v[0]][f[1]] = true;
                }
            } if(f[1] + 1 <= k && dist[v[0]][f[1] + 1] > dist[f[0]][f[1]] + v[1] / 2) {
                dist[v[0]][f[1] + 1] = dist[f[0]][f[1]] + v[1] / 2;
                // printf("from: %d %d to: %d %d dis: %d\n", f[0], f[1], v[0], f[1] + 1, dist[v[0]][f[1] + 1]);
                if(!b[v[0]][f[1] + 1]) {
                    que.push(A{v[0], f[1] + 1});
                    b[v[0]][f[1] + 1] = true;
                }
            }
        }
    }
}
```



 ### kosaraju

```c++
//kosaraju
struct edge{int to, nex;}e[2][maxn << 1];
int dp[maxn], u[maxn], v[maxn];
int head[2][maxn], tot, q[maxn], vis[maxn], idx, f[maxn], n, m, a[maxn];
struct node{int num, v;};
void add(int x, int y) {
    e[0][++ tot].to = y;
    //e[0][tot].dist = d;
    e[0][tot].nex = head[0][x];
    head[0][x] = tot;
    e[1][tot].to = x;
    //e[1][tot].dist = d;
    e[1][tot].nex = head[1][y];
    head[1][y] = tot;
}

void dfs1(int x) {
    vis[x] = 1;
    for(int i = head[0][x]; i; i = e[0][i].nex) {
        if(!vis[e[0][i].to]) dfs1(e[0][i].to);
    }
    q[++ idx] = x;
}


void dfs2(int x, int y) {
    vis[x] = 0; f[x] = y;
    if(y != x) a[y] += a[x];
    for(int i = head[1][x]; i; i = e[1][i].nex) {
        if(vis[e[1][i].to]) dfs2(e[1][i].to, y); 
    } 
}

void dfs3(int x) {
    vis[x] = 1;
    for(int i = head[0][x]; i; i = e[0][i].nex) {
        if(!vis[f[e[0][i].to]]) dfs3(f[e[0][i].to]);
        dp[x] = max(dp[f[e[0][i].to]], dp[x]);
    }
    dp[x] += a[x];
}

int main() {
    cin >> n >> m;
    for(int i = 1; i <= n; i ++) f[i] = i;
    for(int i = 1; i <= n; i ++) cin >> a[i];
    for(int i = 1; i <= m; i ++) {
        cin >> u[i] >> v[i];
        add(u[i], v[i]);
    }
    dfs1(1);
    for(int i = n; i >= 1; i --) 
        if(vis[q[i]]) dfs2(q[i], q[i]);
    
    memset(e, 0, sizeof e);
    memset(head, 0, sizeof head);
    idx = tot = 0;
    for(int i = 1; i <= m; i ++) {
        if(f[u[i]] != f[v[i]]) add(f[u[i]], f[v[i]]);
    }
    memset(vis, 0, sizeof vis);
    for(int i = 1; i <= n; i ++) {
        if(!dp[f[i]]) dfs3(f[i]);
    }
    int res = 0;
    for(int i = 1; i <= n; i ++) {
        res = max(res, dp[f[i]]);
    }
    cout << res << "\n";
    return 0;
}
```

### tarjan(点强连通)

```c++
#include <bits/stdc++.h>
using namespace std;
const int mxn = 2e5 + 10;
const int N = 1e6 + 10;
vector<int> e[mxn];
vector<int> vec[mxn];
vector<vector<int>> scc;
stack<int> stk;
int n, m, mod, idx, cnt; 
int bel[mxn], ins[mxn], dfn[mxn], low[mxn]; 

void tarjan(int u) {
	ins[u] = true;
	dfn[u] = low[u] = ++ idx;
	stk.push(u);
	for(auto v : e[u]) {
		if(!dfn[v]) tarjan(v);
		if(ins[v]) low[u] = min(low[u], low[v]);
	}
	if(dfn[u] == low[u]) {
		++ cnt; 
		while(true) {
			int v = stk.top();
			ins[v] = false;
			stk.pop();
			bel[v] = cnt; 
			vec[cnt].push_back(v);
			if(v == u) break; 
		}
	}
}
```



### 求割边

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 2e5 + 10;
vector<vector<int>> scc;
vector<pair<int, int>> e[N];
int dfn[N],bel[N], low[N], n, m, idx;
vector<int> bridge;
void dfs(int u, int id) {
    dfn[u] = low[u] = ++ idx;
    for(auto pii : e[u]) {
        auto v = pii.first, id2 = pii.second;
        if(!dfn[v]) dfs(v, id2);
        if(id != id2) low[u] = min(low[u], low[v]);
    }
    if(dfn[u] == low[u] && id != -1) {
        bridge.push_back(id);
    }
}
```



### tarjan(点双连通)

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 2e5 + 10;
vector<int> e[N], cc[N];
int bel[N], dfn[N], low[N], cut[N], idx, cnt;
stack<int> stk;
void tarjan(int u, int f) {
    dfn[u] = low[u] = ++ idx;
    int ch = 0;stk.push(u);
    for(auto v : e[u]) {
        if(!dfn[v]) {
            tarjan(v, u);
            ch ++;
            low[u] = min(low[u], low[v]);
            if(low[v] >= dfn[u]) {
                cut[u] = 1;
                ++ cnt;
                cc[cnt].clear();
                cc[cnt].push_back(u);
                while(true) {
                    int w = stk.top();
                    cc[cnt].push_back(w);
                    stk.pop();
                    if(w == v) break;
                }
            }
        }else if(v != f)
                low[u] = min(low[u], dfn[v]);
    }
    if(u == 1 && ch <= 1) cut[u] = 0;
}
int n, m;
int main() {
    int tc = 0;
    while(true) {
        scanf("%d", &m);
        if(m == 0) break;
        n = 0;
        for(int i = 1; i <= 1000; i ++) 
            e[i].clear();
        idx = cnt = 0;
        while(stk.size()) stk.pop();
        for(int i = 1; i <= m; i ++) {
            int u, v;
            scanf("%d%d", &u, &v);
            e[u].push_back(v);
            e[v].push_back(u);
            n = max({n, u, v});
        }
        for(int i = 1; i <= n; i ++)
            dfn[i] = low[i] = cut[i] = 0;
        // tarjan(1, 0);
        tarjan(1, 0);
        printf("Case %d: ", ++ tc);
        if(cnt == 1) {
            // cout << "c1" << '\n';
            n = cc[1].size();
            printf("%d %d\n", 2, 1ll * n * (n - 1) / 2);
        } else {
            int ans1 = 0;
            long long ans2 = 1;
            for(int i = 1; i <= cnt; i ++) {
                int ncut = 0;
                for(auto u : cc[i]) ncut += cut[u];
                if(ncut == 1) {
                    ans1 += 1;
                    ans2 *= (int) cc[i].size() - 1;
                }
            }
            printf("%d %lld\n", ans1, ans2);
        }
    }
}

```

### 网络流

```c++
const int V = 20100;
const int E = 101000;
template<typename T>
struct FlowGraph {
	int s, t, vtot;
	int head[V], etot;
	int dis[V], cur[V];
	struct edge {
		int v, nxt;
		T f;
	} e[E * 2];
	void addedge(int u,int v, T f){
		e[etot]= {v, head[u], f}; head[u] = etot++;
		e[etot]= {u, head[v], 0}; head[v] = etot++;
	}

	bool bfs() {
		for (int i = 1; i <= vtot; i++) {
			dis[i] = 0;
			cur[i] = head[i];
		}
		queue<int> q;
		q.push(s); dis[s] = 1;
		while (!q.empty()) {
			int u = q.front(); q.pop();
			for (int i = head[u]; ~i; i = e[i].nxt) {
				if (e[i].f && !dis[e[i].v]) {
					int v = e[i].v;
					dis[v] = dis[u] + 1;
					if (v == t) return true;
					q.push(v);
				}
			}
		}
		return false;
	}

	T dfs(int u, T m) {
		if (u == t) return m;
		T flow = 0;
		for (int i = cur[u]; ~i; cur[u] = i = e[i].nxt)
			if (e[i].f && dis[e[i].v] == dis[u] + 1) {
				T f = dfs(e[i].v, min(m, e[i].f));
				e[i].f -= f;
				e[i ^ 1].f += f;
				m -= f;
				flow += f;
				if (!m) break;
			}
		if (!flow) dis[u] = -1;
		return flow;
	}
	T dinic(){
		T flow=0;
		while (bfs()) flow += dfs(s, numeric_limits<T>::max());
		return flow;
	}
	void init(int s_, int t_, int vtot_) {
		s = s_;
		t = t_;
		vtot = vtot_;
		etot = 0;
		for (int i = 1; i <= vtot; i++) head[i] = -1;
	}
};

```

### 最小费用流

```c++
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

const int V = 20100;
const int E = 201000;

template<typename T>
struct MinCostGraph {
	int s, t, vtot;
	int head[V], etot;
	T dis[V], flow, cost;
	int pre[V];
	bool vis[V];

	struct edge {
		int v, nxt;
		T f, c;
	} e[E * 2];
	void addedge(int u,int v, T f, T c, T f2 = 0){
		e[etot]= {v, head[u], f, c}; head[u] = etot++;
		e[etot]= {u, head[v], f2, -c}; head[v] = etot++;
	}

	bool spfa() {
		T inf = numeric_limits<T>::max() / 2;
		for (int i = 1; i <= vtot; i++) {
			dis[i] = inf;
			vis[i] = false;
			pre[i] = -1;
		}
		dis[s] = 0;
		vis[s] = true;
		queue<int> q;
		q.push(s);
		while (!q.empty()) {
			int u = q.front();
			for (int i = head[u]; ~i; i = e[i].nxt) {
				int v = e[i].v;
				if (e[i].f && dis[v] > dis[u] + e[i].c) {
					dis[v] = dis[u] + e[i].c;
					pre[v] = i;
					if (!vis[v]) {
						vis[v] = 1;
						q.push(v);
					}
				}
			}
			q.pop();
			vis[u] = false;
		}
		return dis[t] != inf;
	}

	void augment() {
		int u = t;
		T f = numeric_limits<T>::max();
		while (~pre[u]) {
			f = min(f, e[pre[u]].f);
			u = e[pre[u] ^ 1].v;
		}
		flow += f;
		cost += f * dis[t];
		u = t;
		while (~pre[u]) {
			e[pre[u]].f -= f;
			e[pre[u] ^ 1].f += f;
			u = e[pre[u] ^ 1].v;
		}
	}

	pair<T, T> solve() {
		flow = 0;
		cost = 0;
		while (spfa()) augment();
		return {flow, cost};
	}
	void init(int s_, int t_, int vtot_) {
		s = s_;
		t = t_;
		vtot = vtot_;
		etot = 0;
		for (int i = 1; i <= vtot; i++) head[i] = -1;
	}
};

MinCostGraph<int> g;
int n, m;
array<int, 3> e[E];
int main() {
	scanf("%d%d", &n, &m);
	g.init(1, n, n);
	for (int i = 1; i <= m; i++) {
		int u, v, f, c;
		scanf("%d%d%d%d", &u, &v, &f, &c);
		g.addedge(u, v, f, c);
	}
	auto [flow, cost] = g.solve();
	printf("%d %d\n", flow, cost);
}
```

### 二分匹配

$$
hall定理：对于一个二分图存在完备匹配， 当且仅当对于左边\forall S，S的邻居集合的并 N（S）
保证：|N(S)|\geqslant|S|
$$

$$
Dilworth定理：最大反链=最小链覆盖（可相交）
$$



## 数论

### exgcd

```c++
int exgcd(int a, int b, int &x, int &y) {
    if(b == 0) return x = 1, y = 0, a;
    int r = exgcd(b, a % b, x, y);
    tie(x, y) = make_tuple(y, x - (a / b) * y);
    return r;
}
```

### 组合数

```c++
ll fac[N + 10], inv[N + 10];

ll qpow(ll a, ll b) {
    ll res = 1;
    for(; b; b >>= 1) {
        if(b & 1) res = (a * res) % mod;
        a = (a * a) % mod;
    }return res;
}

void init() {
    fac[0] = 1;
    for(int i = 1; i <= N; i ++)
        fac[i] = fac[i - 1] * i % mod;
    inv[N] = qpow(fac[N], mod - 2);
    for(int i = N - 1; i >= 0; i --)
        inv[i] = inv[i + 1] * (i + 1) % mod;
    // cout << inv[1] << '\n';
    assert(inv[0] == 1);
}

int C(int n, int m) {
    if(n < 0 || m > n) return 0;
    return (fac[n] * inv[m] % mod) * inv[n - m] % mod;
}

//数据量较小的话可以预处理组合数
int c[505][505];

void init1(){
    c[0][0]=1;
	for(int i=1;i<=500;i++){
		c[i][0]=1;
		for(int j=1;j<=i;j++)c[i][j]=(c[i-1][j-1]+c[i-1][j])%mod;//预处理组合数 
	}
}
```

### 埃氏筛

```c++
const int Maxn=1e3+10;
int ol[Maxn],prime[Maxn],ans[Maxn];
void init(){
    ol[1]=1;
    for(int i=2;i<=1000;i++){
        if(ol[i])continue;
        prime[++prime[0]]=i;
        for(long long j=1ll*i*i;j<=1000;j+=i){
            ol[j]=1;
        }
    }
}
```

### 欧拉筛

```c++
const int mxn = 1e7;
bool not_prime[mxn + 10];
int prime[mxn + 10], tot;
void ol(int n) {
    for(int i = 2; i <= n; i ++) {
        if(!not_prime[i]) prime[++ tot] = i;
        for(int j = 1; j <= tot && i * prime[j] <= n; j ++) {
            not_prime[i * prime[j]] = 1;
            if(i % prime[j] == 0) break;
        }
    }
}
```

### 欧拉公式

$$
\phi(n) = (p_1 - 1)\cdot p_1^{a_1-1}\cdot(p_2-1)\cdot p_1^{a_2-1}\cdots(p_n - 1)\cdotp_n^{a_n - 1}
$$

```c++
using ll = long long;
ll get_phi(ll n) {
    ll phi = 1;
    for(int i = 2; i <= n / i; i ++) {
        if(n % i == 0) {
            phi *= (i - 1);
            n /= i;
            while(n % i == 0) phi *= i, n /= i;
        }
    }
    if(n > 1) phi *= n - 1;
    return phi;
}//一个数n,求小于n的互质的个数
```

在欧拉筛的基础上求1~n的欧拉函数

```c++
void init(int n) {
    for (int i = 1; i <= n; i++)
        phi[i] = i; // 除1外没有数的欧拉函数是本身，所以如果phi[i] = i则说明未被筛到
    for (int i = 2; i <= n; i++)
        if (phi[i] == i) // 未被筛到
            for (int j = i; j <= n; j += i) // 所有含有该因子的数都进行一次操作
                phi[j] = phi[j] / i * (i - 1);
}
```

### 线性求逆元

```c++
    inv[1] = 1;
    for(int i = 2; i <= n; i ++) {
		inv[i] = (p - p / i) * inv[p % i] % p;

	}
    for(int i = 1; i <= n; i ++) 
        cout << inv[i] << "\n";
    return 0;
```

### 整除分块

```c++
using ull = unsigned long long;
int main() {
	ull n, sum = 0;
	cin >> n;
	for(ull l = 1; l <= n; l ++) {
		ull d = n / l, r = n / d;
		sum += (n - d * l + n - d * r) * (r - l + 1) / 2;
		l = r;
	}
}
```



### 中国剩余定理

```c++
int exgcd(int a, int b, int &x, int &y) {
    if(b == 0) return x = 1, y = 0, a;
    int r = exgcd(b, a % b, x, y);
    tie(x, y) = make_tuple(y, x - (a / b) * y);
    return r;
}

int lcm(int a, int b) {return a / __gcd(a, b) * b;}

int excrt(int k, int *a, int *r) {
    int m = r[1], ans = a[1];
    for(int i = 2; i <= k; i ++) {
        int x0, y0, c = a[i] - ans;
        int g = exgcd(m, r[i], x0, y0);
        if(c % g != 0) return -1;
        x0 = (__int128) x0 * (c / g) % (r[i] / g);
        ans = x0 * m + ans;
        m  = lcm(m, r[i]);
        ans = (ans % m + m) % m;
    }
    return ans;
}
```



### 阶

```c++
int x = p - 1;
for(int i = 1; i <= tot; i ++) {
	while(x % pf[i] == 0 && qpow(a, x / pf[i]) == 1)
	x /= pf[i];
}
```

最小的$x$满足$a^x=1(\mod p)$

### MEX

```c++
int cot[N];
set<int>st;
int m, op, x;
void slove() {
    for (int i = 0; i < N; i++)st.insert(i);
    cin >> m;
    while (m--) {
        cin >> op;
        if (op == 0) {//添加
            cin >> x;
            if (cot[x] == 0) {
                st.erase(x);
            }
            cot[x]++;
        }
        else if (op == 1) {//查询
            cout << *st.begin() << endl;
        }
        else {//删除
            cin >> x;
            if (cot[x] == 1) {
                st.insert(x);
            }
            cot[x]--;
        }
    }
}
```



### 矩阵快速幂

```c++
class Mat
{
    #define inc(x,y) ((x+y>=P)?x+y-P:x+y)
	#define dec(x,y) ((x-y<0)?x-y+P:x-y)
    static const int N=105;
private:
	int a[2][2]={};
public:
//basic declare
    size_t n=0,m=0;
    Mat(){}
	Mat(int n_,int m_):n(n_),m(m_){}
    Mat &operator=(const initializer_list<initializer_list<int>>&t)
    {
        n=t.size(),m=t.begin()->size();
        for(auto it=t.begin();it!=t.end();it++)
            copy(it->begin(),it->end(),a[it-t.begin()]);
        return *this;
    }
    Mat(const initializer_list<initializer_list<int>>&t){*this=t;}
    int *operator[](size_t i){return a[i];}
    const int *operator[](size_t i) const{return a[i];}
//operator
    Mat &operator*=(const Mat &b)
    {
        assert(m==b.n);
        Mat c(n,b.m);
        for(int i=0;i<n;i++)
            for(int j=0;j<b.m;j++)
                for(int k=0;k<m;k++)
                    (c[i][j]+=a[i][k]*b[k][j]%mod)%=mod;
        return *this=c;
    }
    friend Mat operator*(Mat a,const Mat &b){return a*=b;}
    Mat &operator^(long long b)
    {
        assert(n==m);
        Mat c;
        c.identity(n);
        while(b)
        {
            if(b&1) c*=(*this);
            b>>=1,(*this)*=(*this);
        }
        return *this=c;
    }
    void identity(const size_t &t)
    {
        n=m=t;
        for(int i=0;i<n;i++) 
            for(int j=0;j<n;j++)
                a[i][j]=(i==j);
    }
};
```



### FFT快速傅里叶变换

```c++
const db Pi = acos(-1);
int n, m;
struct CP {
    CP (double xx = 0, double yy = 0) {x = xx, y = yy;}
    double x, y;
    CP operator + (CP const &B) const 
    {return CP(x + B.x, y + B.y);}
    CP operator - (CP const &B) const
    {return CP(x - B.x, y - B.y);}
    CP operator * (CP const &B) const 
    {return CP(x * B.x - y * B.y, x * B.y + y * B.x);}
}f[mxn << 1], p[mxn << 1];
int tr[mxn << 1];
void fft(CP *f, bool flag) {
    for(int i = 0; i < n; i ++)
        if(i < tr[i]) swap(f[i], f[tr[i]]);
    for(int p = 2; p <= n; p <<= 1) {
        int len = p >> 1;
        CP tG(cos(2 * Pi / p), sin(2 * Pi / p));
        if(!flag) tG.y *= -1;
        for(int k = 0; k < n;k += p) {
            CP buf(1, 0);
            for(int l = k; l < k + len; l ++) {
                CP tt = buf * f[len + l];
                f[len + l] = f[l] - tt;
                f[l] = f[l] + tt;
                buf = buf * tG;
            }
        }
    }
}
int main() {
    scanf("%d%d", &n, &m);
    for(int i = 0; i <= n; i ++) scanf("%lf", &f[i].x);
    for(int i = 0; i <= m; i ++) scanf("%lf", &p[i].x);
    for(m += n, n = 1; n <= m; n <<= 1);
    for(int i = 0; i < n; i ++)
        tr[i] = (tr[i >> 1] >> 1) | ((i & 1) ? n >> 1 : 0);//蝴蝶变换  
    fft(f, 1); fft(p, 1);
    for(int i = 0; i < n; i ++) f[i] = f[i] * p[i];
    fft(f, 0);
    for(int i = 0; i <= m; i ++) printf("%d ", (int) (f[i].x / n + 0.49));
    return 0;
}

```

### fastfft

````c++
// FFT_MAXN = 2^k
// fft_init() to precalc FFT_MAXN-th roots

typedef long double db;
const int FFT_MAXN=262144;
const db pi=acos(-1.);
struct cp{
	db a,b;
	cp operator+(const cp&y)const{return (cp){a+y.a,b+y.b};}
	cp operator-(const cp&y)const{return (cp){a-y.a,b-y.b};}
	cp operator*(const cp&y)const{return (cp){a*y.a-b*y.b,a*y.b+b*y.a};}
	cp operator!()const{return (cp){a,-b};};
}nw[FFT_MAXN+1];int bitrev[FFT_MAXN];
void dft(cp*a,int n,int flag=1){
	int d=0;while((1<<d)*n!=FFT_MAXN)d++;
	rep(i,0,n)if(i<(bitrev[i]>>d))swap(a[i],a[bitrev[i]>>d]);
	for (int l=2;l<=n;l<<=1){
		int del=FFT_MAXN/l*flag;
		for (int i=0;i<n;i+=l){
			cp *le=a+i,*ri=a+i+(l>>1),*w=flag==1?nw:nw+FFT_MAXN;
			rep(k,0,l>>1){
				cp ne=*ri**w;
				*ri=*le-ne,*le=*le+ne;
				le++,ri++,w+=del;
			}
		}
	}
	if(flag!=1)rep(i,0,n)a[i].a/=n,a[i].b/=n;
}
void fft_init(){
	int L=0;while((1<<L)!=FFT_MAXN)L++;
	bitrev[0]=0;rep(i,1,FFT_MAXN)bitrev[i]=bitrev[i>>1]>>1|((i&1)<<(L-1));
	nw[0]=nw[FFT_MAXN]=(cp){1,0};
	rep(i,0,FFT_MAXN+1)nw[i]=(cp){cosl(2*pi/FFT_MAXN*i),sinl(2*pi/FFT_MAXN*i)};	//very slow
}

void convo(db*a,int n,db*b,int m,db*c){
	static cp f[FFT_MAXN>>1],g[FFT_MAXN>>1],t[FFT_MAXN>>1];
	int N=2;while(N<=n+m)N<<=1;
	rep(i,0,N)
		if(i&1){
			f[i>>1].b=(i<=n)?a[i]:0.0;
			g[i>>1].b=(i<=m)?b[i]:0.0;
		}else{
			f[i>>1].a=(i<=n)?a[i]:0.0;
			g[i>>1].a=(i<=m)?b[i]:0.0;
		}
	dft(f,N>>1);dft(g,N>>1);
	int del=FFT_MAXN/(N>>1);
	cp qua=(cp){0,0.25},one=(cp){1,0},four=(cp){4,0},*w=nw;
	rep(i,0,N>>1){
		int j=i?(N>>1)-i:0;
		t[i]=(four*!(f[j]*g[j])-(!f[j]-f[i])*(!g[j]-g[i])*(one+*w))*qua;
		w+=del;
	}
	dft(t,N>>1,-1);
	rep(i,0,n+m+1)c[i]=(i&1)?t[i>>1].a:t[i>>1].b;
}

void mul(int *a,int *b,int n){// n<=N, 0<=a[i],b[i]<mo
	static cp f[N],g[N],t[N],r[N];
	int nn=2;while(nn<=n+n)nn<<=1;
	rep(i,0,nn){
		f[i]=(i<=n)?(cp){(db)(a[i]>>15),(db)(a[i]&32767)}:(cp){0,0};
		g[i]=(i<=n)?(cp){(db)(b[i]>>15),(db)(b[i]&32767)}:(cp){0,0};
	}
	swap(n,nn);
	dft(f,n,1);dft(g,n,1);
	rep(i,0,n){
		int j=i?n-i:0;
		t[i]=( (f[i]+!f[j])*(!g[j]-g[i]) + (!f[j]-f[i])*(g[i]+!g[j]) )*(cp){0,0.25};
		r[i]=(!f[j]-f[i])*(!g[j]-g[i])*(cp){-0.25,0} + (cp){0,0.25}*(f[i]+!f[j])*(g[i]+!g[j]);
	}
	dft(t,n,-1); dft(r,n,-1);
	rep(i,0,n)a[i]=( (ll(t[i].a+0.5)%mo<<15) + ll(r[i].a+0.5) + (ll(r[i].b+0.5)%mo<<30) )%mo;
}

````

### ntt

```c++
const int md = 998244353;

inline void add(int &x, int y) {
  x += y;
  if (x >= md) {
    x -= md;
  }
}

inline void sub(int &x, int y) {
  x -= y;
  if (x < 0) {
    x += md;
  }
}

inline int mul(int x, int y) {
  return (long long) x * y % md;
}

inline int power(int x, int y) {
  int res = 1;
  for (; y; y >>= 1, x = mul(x, x)) {
    if (y & 1) {
      res = mul(res, x);
    }
  }
  return res;
}

inline int inv(int a) {
  a %= md;
  if (a < 0) {
    a += md;
  }
  int b = md, u = 0, v = 1;
  while (a) {
    int t = b / a;
    b -= t * a;
    swap(a, b);
    u -= t * v;
    swap(u, v);
  }
  if (u < 0) {
    u += md;
  }
  return u;
}

namespace ntt {
int base = 1, root = -1, max_base = -1;
vector<int> rev = {0, 1}, roots = {0, 1};

void init() {
  int temp = md - 1;
  max_base = 0;
  while (temp % 2 == 0) {
    temp >>= 1;
    ++max_base;
  }
  root = 2;
  while (true) {
    if (power(root, 1 << max_base) == 1 && power(root, 1 << (max_base - 1)) != 1) {
      break;
    }
    ++root;
  }
}

void ensure_base(int nbase) {
  if (max_base == -1) {
    init();
  }
  if (nbase <= base) {
    return;
  }
  assert(nbase <= max_base);
  rev.resize(1 << nbase);
  for (int i = 0; i < 1 << nbase; ++i) {
    rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (nbase - 1));
  }
  roots.resize(1 << nbase);
  while (base < nbase) {
    int z = power(root, 1 << (max_base - 1 - base));
    for (int i = 1 << (base - 1); i < 1 << base; ++i) {
      roots[i << 1] = roots[i];
      roots[i << 1 | 1] = mul(roots[i], z);
    }
    ++base;
  }
}

void dft(vector<int> &a) {
  int n = a.size(), zeros = __builtin_ctz(n);
  ensure_base(zeros);
  int shift = base - zeros;
  for (int i = 0; i < n; ++i) {
    if (i < rev[i] >> shift) {
      swap(a[i], a[rev[i] >> shift]);
    }
  }
  for (int i = 1; i < n; i <<= 1) {
    for (int j = 0; j < n; j += i << 1) {
      for (int k = 0; k < i; ++k) {
        int x = a[j + k], y = mul(a[j + k + i], roots[i + k]);
        a[j + k] = (x + y) % md;
        a[j + k + i] = (x + md - y) % md;
      }
    }
  }
}

vector<int> multiply(vector<int> a, vector<int> b) {
  int need = a.size() + b.size() - 1, nbase = 0;
  while (1 << nbase < need) {
    ++nbase;
  }
  ensure_base(nbase);
  int sz = 1 << nbase;
  a.resize(sz);
  b.resize(sz);
  bool equal = a == b;
  dft(a);
  if (equal) {
    b = a;
  } else {
    dft(b);
  }
  int inv_sz = inv(sz);
  for (int i = 0; i < sz; ++i) {
    a[i] = mul(mul(a[i], b[i]), inv_sz);
  }
  reverse(a.begin() + 1, a.end());
  dft(a);
  a.resize(need);
  return a;
}

vector<int> inverse(vector<int> a) {
  int n = a.size(), m = (n + 1) >> 1;
  if (n == 1) {
    return vector<int>(1, inv(a[0]));
  } else {
    vector<int> b = inverse(vector<int>(a.begin(), a.begin() + m));
    int need = n << 1, nbase = 0;
    while (1 << nbase < need) {
      ++nbase;
    }
    ensure_base(nbase);
    int sz = 1 << nbase;
    a.resize(sz);
    b.resize(sz);
    dft(a);
    dft(b);
    int inv_sz = inv(sz);
    for (int i = 0; i < sz; ++i) {
      a[i] = mul(mul(md + 2 - mul(a[i], b[i]), b[i]), inv_sz);
    }
    reverse(a.begin() + 1, a.end());
    dft(a);
    a.resize(n);
    return a;
  }
}
}

using ntt::multiply;
using ntt::inverse;

vector<int>& operator += (vector<int> &a, const vector<int> &b) {
  if (a.size() < b.size()) {
    a.resize(b.size());
  }
  for (int i = 0; i < b.size(); ++i) {
    add(a[i], b[i]);
  }
  return a;
}

vector<int> operator + (const vector<int> &a, const vector<int> &b) {
  vector<int> c = a;
  return c += b;
}

vector<int>& operator -= (vector<int> &a, const vector<int> &b) {
  if (a.size() < b.size()) {
    a.resize(b.size());
  }
  for (int i = 0; i < b.size(); ++i) {
    sub(a[i], b[i]);
  }
  return a;
}

vector<int> operator - (const vector<int> &a, const vector<int> &b) {
  vector<int> c = a;
  return c -= b;
}

vector<int>& operator *= (vector<int> &a, const vector<int> &b) {
  if (min(a.size(), b.size()) < 128) {
    vector<int> c = a;
    a.assign(a.size() + b.size() - 1, 0);
    for (int i = 0; i < c.size(); ++i) {
      for (int j = 0; j < b.size(); ++j) {
        add(a[i + j], mul(c[i], b[j]));
      }
    }
  } else {
    a = multiply(a, b);
  }
  return a;
}

vector<int> operator * (const vector<int> &a, const vector<int> &b) {
  vector<int> c = a;
  return c *= b;
}

vector<int>& operator /= (vector<int> &a, const vector<int> &b) {
  int n = a.size(), m = b.size();
  if (n < m) {
    a.clear();
  } else {
    vector<int> c = b;
    reverse(a.begin(), a.end());
    reverse(c.begin(), c.end());
    c.resize(n - m + 1);
    a *= inverse(c);
    a.erase(a.begin() + n - m + 1, a.end());
    reverse(a.begin(), a.end());
  }
  return a;
}

vector<int> operator / (const vector<int> &a, const vector<int> &b) {
  vector<int> c = a;
  return c /= b;
}

vector<int>& operator %= (vector<int> &a, const vector<int> &b) {
  int n = a.size(), m = b.size();
  if (n >= m) {
    vector<int> c = (a / b) * b;
    a.resize(m - 1);
    for (int i = 0; i < m - 1; ++i) {
      sub(a[i], c[i]);
    }
  }
  return a;
}

vector<int> operator % (const vector<int> &a, const vector<int> &b) {
  vector<int> c = a;
  return c %= b;
}

vector<int> derivative(const vector<int> &a) {
  int n = a.size();
  vector<int> b(n - 1);
  for (int i = 1; i < n; ++i) {
    b[i - 1] = mul(a[i], i);
  }
  return b;
}

vector<int> primitive(const vector<int> &a) {
  int n = a.size();
  vector<int> b(n + 1), invs(n + 1);
  for (int i = 1; i <= n; ++i) {
    invs[i] = i == 1 ? 1 : mul(md - md / i, invs[md % i]);
    b[i] = mul(a[i - 1], invs[i]);
  }
  return b;
}

vector<int> logarithm(const vector<int> &a) {
  vector<int> b = primitive(derivative(a) * inverse(a));
  b.resize(a.size());
  return b;
}

vector<int> exponent(const vector<int> &a) {
  vector<int> b(1, 1);
  while (b.size() < a.size()) {
    vector<int> c(a.begin(), a.begin() + min(a.size(), b.size() << 1));
    add(c[0], 1);
    vector<int> old_b = b;
    b.resize(b.size() << 1);
    c -= logarithm(b);
    c *= old_b;
    for (int i = b.size() >> 1; i < b.size(); ++i) {
      b[i] = c[i];
    }
  }
  b.resize(a.size());
  return b;
}

vector<int> power(vector<int> a, int m) {
  int n = a.size(), p = -1;
  vector<int> b(n);
  for (int i = 0; i < n; ++i) {
    if (a[i]) {
      p = i;
      break;
    }
  }
  if (p == -1) {
    b[0] = !m;
    return b;
  }
  if ((long long) m * p >= n) {
    return b;
  }
  int mu = power(a[p], m), di = inv(a[p]);
  vector<int> c(n - m * p);
  for (int i = 0; i < n - m * p; ++i) {
    c[i] = mul(a[i + p], di);
  }
  c = logarithm(c);
  for (int i = 0; i < n - m * p; ++i) {
    c[i] = mul(c[i], m);
  }
  c = exponent(c);
  for (int i = 0; i < n - m * p; ++i) {
    b[i + m * p] = mul(c[i], mu);
  }
  return b;
}

vector<int> sqrt(const vector<int> &a) {
  vector<int> b(1, 1);
  while (b.size() < a.size()) {
    vector<int> c(a.begin(), a.begin() + min(a.size(), b.size() << 1));
    vector<int> old_b = b;
    b.resize(b.size() << 1);
    c *= inverse(b);
    for (int i = b.size() >> 1; i < b.size(); ++i) {
      b[i] = mul(c[i], (md + 1) >> 1);
    }
  }
  b.resize(a.size());
  return b;
}

vector<int> multiply_all(int l, int r, vector<vector<int>> &all) {
  if (l > r) {
    return vector<int>();
  } else if (l == r) {
    return all[l];
  } else {
    int y = (l + r) >> 1;
    return multiply_all(l, y, all) * multiply_all(y + 1, r, all);
  }
}

vector<int> evaluate(const vector<int> &f, const vector<int> &x) {
  int n = x.size();
  if (!n) {
    return vector<int>();
  }
  vector<vector<int>> up(n * 2);
  for (int i = 0; i < n; ++i) {
    up[i + n] = vector<int>{(md - x[i]) % md, 1};
  }
  for (int i = n - 1; i; --i) {
    up[i] = up[i << 1] * up[i << 1 | 1];
  }
  vector<vector<int>> down(n * 2);
  down[1] = f % up[1];
  for (int i = 2; i < n * 2; ++i) {
    down[i] = down[i >> 1] % up[i];
  }
  vector<int> y(n);
  for (int i = 0; i < n; ++i) {
    y[i] = down[i + n][0];
  }
  return y;
}

vector<int> interpolate(const vector<int> &x, const vector<int> &y) {
  int n = x.size();
  vector<vector<int>> up(n * 2);
  for (int i = 0; i < n; ++i) {
    up[i + n] = vector<int>{(md - x[i]) % md, 1};
  }
  for (int i = n - 1; i; --i) {
    up[i] = up[i << 1] * up[i << 1 | 1];
  }
  vector<int> a = evaluate(derivative(up[1]), x);
  for (int i = 0; i < n; ++i) {
    a[i] = mul(y[i], inv(a[i]));
  }
  vector<vector<int>> down(n * 2);
  for (int i = 0; i < n; ++i) {
    down[i + n] = vector<int>(1, a[i]);
  }
  for (int i = n - 1; i; --i) {
    down[i] = down[i << 1] * up[i << 1 | 1] + down[i << 1 | 1] * up[i << 1];
  }
  return down[1];
}

```



### 行列式

```c++
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int N = 210;
int n, m, mod;
ll g[N][N];

int calc(int n) {
    ll ans = 1;
    for(int i = 1; i <= n; i ++)
        for(int j = 1; j <= n; j ++)
            g[i][j] %= mod;
    for(int i = 1; i <= n; i ++) {
        for(int j = i + 1; j <= n; j ++) {
            int x = i, y = j;
            while(g[x][i]) {
                int t = g[y][i] / g[x][i];
                for(int k = i; k <= n; k ++)
                    g[y][k] = (g[y][k] - t * g[x][k]) % mod;
                swap(x, y);
            }
            if(x == i) {
                for(int k = i; k <= n; k ++)
                    swap(g[i][k], g[j][k]);
                ans = - ans;
            }
        }
        if(g[i][i] == 0) {ans = 0; return ans;}
        ans = ans * g[i][i] % mod;
    }
    if(ans < 0) ans += mod;
    return ans;
}

int main() {
    scanf("%d%d%d", &n, &m, &mod);    
    for(int i = 0; i < m; i ++) {
        int u, v; scanf("%d%d", &u, &v);
        g[u][v] --, g[v][u] --;
        g[u][u] ++, g[v][v] ++;
    }
    printf("%d\n", calc(n - 1));
}
```

### 高斯消元

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 510;
int n;
bitset<N << 1> a[N];
int main() {
	scanf("%d", &n);
	for(int i = 1; i <= n; i ++) {
		for(int j = 1; j <= n; j ++) {
			int x; scanf("%d", &x);
			if(x) a[i].set(j);
		}
	}
	for(int i = 1; i <= n; i ++) {
		a[i][n + i] = 1;
	}
	for(int i = 1; i <= n; i ++) {
		for(int j = i + 1; j <= n; j ++) {
			if(a[j][i] == 0) continue;
			if(a[i][i] == 0) swap(a[i], a[j]);
			else a[j] ^= a[i];
		}
	}
	for(int i = n; i >= 1; i --) if(a[i][i] == 1) {
		for(int j = i + 1; j <= n; j ++) {
			if(a[i][j] == 1) a[i] ^= a[j];
		}
	}
	// for(int i = 1; i <= n; i ++) {
	// 	for(int j = 1; j <= n * 2; j ++)
	// 		printf("%d ", a[i][j]);
	// 	puts("");
	// }
	for(int i = 1; i <= n; i ++) {
		if(a[i][i] == 0) {printf("-1\n"); return 0;}
	}
	for(int i = 1; i <= n; i ++) {
		for(int j = 1 + n; j <= n * 2; j ++)
			if(a[i][j] == 1) printf("%d ", j - n);
		puts("");
	}
}
```



### 线性基

```c++
struct linear_basis {
    ll num[B + 1];
    bool insert(ll x) {
        for(int i = B - 1; i >= 0; i --) {
            if(x & (1ll << i)) {
                if(num[i] == 0) {num[i] = x; return true;}
                x ^= num[i];
            }
        }
        return false;
    }
    ll querymin(ll x) {
        for(int i = B - 1; i >= 0; i --) {
            x = min(x, x ^ num[i]);
        }return x;
    }
    ll querymax(ll x) {
        for(int i = B - 1; i >= 0; i --) {
            x = min(x, x ^ num[i]);
        }return x;
    }
}T;
```

### 康托展开

$$
\sum_{i = 1}^{n} sum_{a[i]} \times (n - i) !
$$



## 基础算法

### 快读

```C++
void read(){}
template<typename T,typename... Ts>
inline void read(T &arg,Ts&... args)
{
	T x=0,f=1;
	char c=getchar();
	while(!isdigit(c)){if(c=='-') f=-1;c=getchar();}
	while(isdigit(c)){x=(x<<3)+(x<<1)+(c^'0');c=getchar();}
	arg=x*f;
	read(args...);
}
```



### 快速幂

```c++
inline int fpow(int x,int b){
    int r=1;
    for(x%=mod;b;b>>=1){
        if(b&1)r=r*x%mod;
        x=x*x%mod;
    }
    return r;
}
```

### 字符串暴力/BFS

```c++
int main(void){
  int m;
  cin >> m;
  vector<int> G[10];
  int u, v;
  for(int i = 1; i <= m; i++){
    cin >> u >> v;
    G[u].push_back(v);
    G[v].push_back(u);
  }
  int p; string s = "999999999";
  for(int i = 1; i <= 8; i++){
    cin >> p;
    s[p-1] = i + '0';
  }
  queue<string> Q;
  Q.push(s);
  map<string, int> mp;
  mp[s] = 0;
  while(Q.size()){
    string s = Q.front(); Q.pop();
    for(int i = 1; i <= 9; i++) if(s[i-1] == '9') v = i;
    for(auto u : G[v]){
      string t = s;
      swap(t[u-1], t[v-1]);
      if(mp.count(t)) continue;
      mp[t] = mp[s] + 1;
      Q.push(t);
    }
  }
  if(mp.count("123456789") == 0) cout << -1 << endl;
  else cout << mp["123456789"] << endl;
  return 0;
}
```

### DFS暴力递归

```c++
const int maxn=1e3+550;
const int dir_x[]={1,-1,0,0};
const int dir_y[]={0,0,-1,1};
int vis[maxn][maxn][3];
bool ans;
bool G[maxn][maxn];
int X0,Y0,m,n;

void dfs(int x,int y,int lx,int ly){
	if(ans)return;
	if(vis[x][y][0]&&(vis[x][y][1]!=lx||vis[x][y][2]!=ly)){
		ans=true;
		return;
	}
	vis[x][y][0]=1;vis[x][y][1]=lx;vis[x][y][2]=ly;
	for(int i=0;i<4;i++){
		int xx=(x+dir_x[i]+m+1)%(m+1);
		if(!xx&&dir_x[i]==1)xx++;
		else if(!xx&&dir_x[i]==-1)xx=m;
		int yy=(y+dir_y[i]+n+1)%(n+1);
		if(!yy&&dir_y[i]==1)yy++;
		else if(!yy&&dir_y[i]==-1)yy=n;
		int lly=ly+dir_y[i];
		int llx=lx+dir_x[i];
		if(G[xx][yy]){
			if(!vis[xx][yy][0]||vis[xx][yy][1]!=llx||vis[xx][yy][2]!=lly)
				dfs(xx,yy,llx,lly);
		}
	}
}

```

### 暴力全排列问题

```c++

void prime_permutaion(int n,int *A,int cur){
    if(cur==n&&!ol[A[0]+A[n-1]]){
        for(int i=0;i<n-1;i++){
            cout<<A[i]<<" ";
        }cout<<A[n-1]<<endl;
        return;
    }
    for(int i=1;i<=n;i++){
        int ok=0;
        for(int j=0;j<cur;j++){
            if(A[j]==i){ok=1;break;}
        }
        if(ok)continue;
        if(!ol[A[cur-1]+i]){
            A[cur]=i;
            prime_permutaion(n,A,cur+1);
        }
    }
}
```

### flood_fill

```c++
void dfs(int r,int c,int id){
    if(r<0||r>m||c<0||c>=n)return;
    if(idx[r][c]>0||pic[r][c]!='@')return;
    idx[r][c]=id;
//printf("%d %d %d",r,c,idx[r][c]);
    for(int dr=-1;dr<=1;dr++)
        for(int dc=-1;dc<=1;dc++)
            if(dr!=0||dc!=0)dfs(r+dr,c+dc,id);
}
```



## 数据结构

### 并查集

```c++
const int maxn=1e4+10;
int pa[maxn];
int findset(int x){return pa[x]!=x?pa[x]=findset(pa[x]):x;}
```

### 带启发式合并的并查集

```c++
const int maxn =2e5+10;
int fa[maxn],sizes[maxn];
inline int find(int x){return x==fa[x]?x:fa[x]=find(fa[x]);}
void merge(int u,int v){
    u=find(u),v=find(v);
    if(u==v)return;
    if(sizes[u]<sizes[v])swap(u,v);
    fa[u]=v;
    sizes[v]+=u;
}

```

### st表

```cpp
const int maxn = 1e6 + 10;
int Max[maxn][21];
int query(int l, int r) {
	int k = log2(r - l + 1);
	return max(Max[l][k], Max[r - (1 << k) + 1][k]);
}

int main() {
	for(int i = 1; i <= n; i ++) cin >> Max[i][0];
	for(int j = 1; j <= 21; j ++) 
		for(int i = 1; i + (1 << j) - 1 <= n; i ++)
			Max[i][j] = max(Max[i][j - 1], Max[i + (1 << (j - 1))][j - 1]);
}
```

### 单调队列

```cpp
int front = 1, rear = 0, ans = 0, Left = 1;
for (int i = 1; i <= n; i++) {
    for (; front <= rear && Queue[rear].first < a[i]; rear--);
    Queue[++rear] = {a[i], i};//单调队列里的a[i]是递增的
    for (; front <= rear && Queue[front].first > b[i];) {
        Left++;
        if (Left > Queue[front].second)
            front++;
    }
    ans = max(ans, i - Left + 1);
}
```



### 倍增法求LCA

```c++
const int maxn = 5e5 + 10;
struct zex {int t, nex;}e[maxn << 1 | 1];
int head[maxn], tot;

void AddEdge(int x, int y) {
	e[++ tot].t = y;
	e[tot].nex = head[x];
	head[x] = tot;
}

int depth[maxn], fa[maxn][21], lg[maxn];

void dfs(int now, int fath) {
	fa[now][0] = fath; depth[now] = depth[fath] + 1;
	for(int i = 1; i <= lg[depth[now]]; i ++) 
		fa[now][i] = fa[fa[now][i - 1]][i - 1];
	for(int i = head[now]; i; i = e[i].nex)
		if(e[i].t != fath) dfs(e[i].t, now);
}

int LCA(int x, int y) {
	if(depth[x] < depth[y]) swap(x, y);
	while(depth[x] > depth[y]) 
		x = fa[x][lg[depth[x] - depth[y]] - 1];
	if(x == y) return x;
	for(int k = lg[depth[x]] - 1; k >= 0; k --) 
		if(fa[x][k] != fa[y][k])
			x = fa[x][k], y = fa[y][k];
	return fa[x][0];
}

```

### 树链剖分求LCA

```c++
void dfs1(int x, int fath) {
	sizes[x] = 1; dep[x] = dep[fath] + 1;
	son[x] = 0; fa[x] = fath;
	for(int i = head[x]; i; i = e[i].nex) {
		if(e[i].t == fath) continue;
		dfs1(e[i].t, x);
		sizes[x] += sizes[e[i].t];
		if(sizes[son[x]] < sizes[e[i].t]) son[x] = e[i].t;
	}
}

void dfs2(int x, int tops) {
	top[x] = tops;
	if(son[x]) dfs2(son[x], tops);
	for(int i = head[x]; i; i = e[i].nex) {
		if(e[i].t != fa[x] && e[i].t != son[x]) 
			dfs2(e[i].t, e[i].t);
	}
}

int LCA(int x, int y) {
	while(top[x] != top[y]) {
		if(dep[top[x]] < dep[top[y]]) swap(x, y);
		x = fa[top[x]];
	}
	return dep[x] < dep[y] ? x : y;
}

```



### 线段树

```c++
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int N=1e5+5,mod=1e9+7;
class SegmentTree{
private:
    int n,*res,*tag;
    void pushup(int k){res[k]=(res[k<<1]+res[k<<1|1])%mod;}
    void pushdown(int k,int l,int r){
        int mid=l+r>>1;
        (res[k<<1]+=(mid-l+1)*tag[k]%mod)%=mod;
        (res[k<<1|1]+=(r-mid)*tag[k]%mod)%=mod;
        (tag[k<<1]+=tag[k])%=mod,(tag[k<<1|1]+=tag[k])%=mod;
        tag[k]=0;
    }
    void update(int k,int l,int r,int x,int y,int z){
        if(y<l||x>r) return;
        if(x<=l&&r<=y)
        {
            (res[k]+=(r-l+1)*z%mod)%=mod;
            (tag[k]+=z)%=mod;
            return ;
        }
        pushdown(k,l,r);
        int mid=l+r>>1;
        update(k<<1,l,mid,x,y,z);
        update(k<<1|1,mid+1,r,x,y,z);
        pushup(k);
    }
    int query(int k,int l,int r,int x,int y){
        if(y<l||x>r) return 0;
        if(x<=l&&r<=y) return res[k];
        pushdown(k,l,r);
        int mid=l+r>>1;
        return (query(k<<1,l,mid,x,y)+query(k<<1|1,mid+1,r,x,y))%mod;
    }
public:
    SegmentTree(int size){
        n=size;
        res=new int[(n<<2)+5]();
        tag=new int[(n<<2)+5]();
    }
    ~SegmentTree(){
        delete [] res;
        delete [] tag;
    }
    void update(int l,int r,int w){update(1,1,n,l,r,w);}
    int query(int l,int r){return query(1,1,n,l,r);}
};

```

### xudyh版线段树

```c++
#include <bits/stdc++.h>
using namespace std;
const int mxn = 2e5 + 10;
struct info{int minv, mincnt;};
info operator + (const info& l, const info& r) {
    info a;
    a.minv = min(l.minv, r.minv);
    if(l.minv == r.minv) a.mincnt = l.mincnt + r.mincnt;
    else if(l.minv < r.minv) a.mincnt = l.mincnt;
    else a.mincnt = r.mincnt;
    return a;
}

struct node {info val;}seg[mxn << 2];
int a[mxn];
void pushup(int k) {
    seg[k].val = seg[k << 1].val + seg[k << 1 | 1].val;
}

void build(int k, int l, int r) {
    if(l == r) {seg[k].val = {a[l], 1}; return;}
    int mid = (l + r) >> 1;
    build(k << 1, l, mid), build(k << 1 | 1, mid + 1, r);
    pushup(k);
}

void modify(int k, int l, int r, int pos, int d) {
    if(l == r) {seg[k].val = {d, 1};return;}
    else {
        int mid = (l + r) >> 1;
        if(pos <= mid) modify(k << 1, l, mid, pos, d);
        else if(pos > mid) modify(k << 1 | 1, mid + 1, r, pos, d);
    }
    pushup(k);
}

info query(int k, int l, int r, int ql, int qr) {
    if(ql == l && qr == r) return seg[k].val;
    int mid = (l + r) >> 1;
    if(qr <= mid) return query(k << 1, l, mid, ql, qr);
    else if(ql > mid) return query(k << 1| 1, mid + 1, r, ql, qr);
    return query(k << 1, l, mid, ql, mid) + query(k << 1 | 1, mid + 1, r, mid + 1, qr);
}

int main() {
    int n, q;
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    cin >> n >> q;
    for(int i = 1; i <= n; i ++) cin >> a[i];
    build(1, 1, n);
    for(int i = 1; i <= q; i ++) {
        int opts; cin >> opts;
        if(opts == 1) {
            int x, d; cin >> x >> d;
            modify(1, 1, n, x, d);
        } else {
            int l, r; cin >> l >> r;
            info t = query(1, 1, n, l, r);
            cout << t.minv << ' ' << t.mincnt << '\n';
        }
    }
    return 0;
}
```



### 树状数组

```c++
const int maxn=5e5+10;
int tree[maxn];
inline int lowbit(int x){return x&(-x);}
void update(int x,int k){
    while(x<=500000){
        tree[x]+=k;
        x+=lowbit(x);
    }
}
int query(int x){
    int sum=0;
    while(x){
        sum+=tree[x];
        x-=lowbit(x);
    }
    return sum;
}
```

```c++
template<class T>
struct BIT {
	T c[N];
	int size;
	void init(int s) {
		size = s;
		for (int i = 1; i <= size; i++) c[i] = 0;
	}
	T query(int x) { // 1 ... x
		assert(x <= size);
		T s = 0;
		for (; x; x -= x & (-x)) {
			s += c[x];
		}
		return s;
	}

	void modify(int x, T s) { // a[x] += s
		assert(x != 0);
		for (; x <= size; x += x & (-x)) {
			c[x] += s;
		}
	} 
};
```



## 高维树状数组

```c++
ll c[mxn][mxn], a[mxn][mxn];
int n, m, q;

inline int lowbit(int x){return x & (-x);}

ll query(int x, int y) {
    ll s = 0;
    for(int p = x; p; p -= lowbit(p)) {
        for(int q = y; q; q -= lowbit(q)) {
            s += c[p][q];
        }
    }return s;
}

void modify(int x, int y, ll d) {
    for(int p = x; p <= n; p += lowbit(p)) {
        for(int q = y; q <= m; q += lowbit(q)) {
            c[p][q] += d; 
        }
    }
}
```

### 笛卡尔树

```cpp
void build() {
	stack<int> st;
	int root = 0;
	for (int i = 1; i < n; i++) {
		int last = 0;
		while (!st.empty() && ht[st.top()] > ht[i]) {
			last = st.top();
			st.pop();
		}
		if (!st.empty()) r[st.top()] = i;
		else root = i;
		l[i] = last;
		st.push(i);
	}
	dfs(root, 1, n - 1);
}
```



### 数组链表

```c++
const int Maxn=1e5+10;
int next[Maxn];
char s[Maxn];
int cur,last;
int main()
{
    while(scanf("%s",s+1)==1){
        int len=strlen(s+1);
        cur=last=0;
        next[0]=0;
        for(int i=1;i<=len;i++){
            char ch=s[i];
            if(ch=='[')cur=0;
            else if(ch==']')cur=last;
            else{
                next[i]=next[cur];
                next[cur]=i;
                if(cur==last)last=i;
                cur=i;
            }
        }
        for(int i=next[0];i!=0;i=next[i]){
            printf("%c",s[i]);
        }
        printf("\n");
    }
    return 0;
}
```

### LINK_CUT_TREE

```c++
class LinkCutTree
{
#define ls(k) (ch[0][k])
#define rs(k) (ch[1][k])
private:
    int *ch[2],*s,*f,*r,*v;
    void flip(int x){r[x]^=1;swap(ls(x),rs(x));}
    void pushup(int k){s[k]=s[ls(k)]^s[rs(k)]^v[k];}
    bool chk(int x){return rs(f[x])==x;}
    bool get(int x){return rs(f[x])==x||ls(f[x])==x;}
    void pushdown(int x)
    {
        if(!r[x]) return;
        if(ls(x)) flip(ls(x));
        if(rs(x)) flip(rs(x));
        r[x]=0;
    }
    void pushadd(int x)
    {
        if(get(x)) pushadd(f[x]);
        pushdown(x);
    }
    void rotate(int x)
    {
        int y=f[x],z=f[y],k=chk(x);
        ch[k][y]=ch[k^1][x],f[ch[k^1][x]]=y;
        f[x]=z;if(get(y))ch[chk(y)][z]=x;
        ch[k^1][x]=y,f[y]=x;
        pushup(y),pushup(x);
    }
    void splay(int x)
    {
        pushadd(x);
        while(get(x))
        {
            int y=f[x];
            if(get(y)) chk(x)!=chk(y)?rotate(x):rotate(y);
            rotate(x);
        }
    }
    void access(int x)
    {
        for(int y=0;x;y=x,x=f[x])
            splay(x),rs(x)=y,pushup(x);
    }
    void makert(int x)
    {
        access(x);
        splay(x);
        flip(x);
    }
    int findrt(int x)
    {
        access(x),splay(x);
        pushdown(x);
        while(ls(x)) x=ls(x),pushdown(x);
        return x;
    }
    void split(int x,int y)
    {
        makert(x);
        access(y);
        splay(y);
    }
public:
    LinkCutTree(int n)
    {
        int size=n+5;
        ch[0]=new int[size]();
        ch[1]=new int[size]();
        s=new int[size]();
        f=new int[size]();
        r=new int[size]();
        v=new int[size]();
    }
    ~LinkCutTree()
    {
        delete [] ch[0];
        delete [] ch[1];
        delete [] s;
        delete [] f;
        delete [] r;
        delete [] v;
    }
    void link(int x,int y)
    {
        makert(x);
        if(findrt(y)!=x) f[x]=y;
    }
    void cut(int x,int y)
    {
        split(x,y);
        if(findrt(y)==x&&!rs(x)&&f[x]==y)
        {
            ls(y)=f[x]=0;
            pushup(y);
        }
    }
    int query(int x,int y)//返回x到y的异或和
    {
        split(x,y);
        return s[y];
    }
    void update(int x,int w)
    {
        splay(x),v[x]=w,pushup(x);
    }
};
```

### 哈希表

```c++
const int hashsize=1000003;
int head[hashsize],next[maxstate];
void init_lookup_table(){memset(head,0,sizeof(head));}
int hash(State& s){
    int v=0;
    for(int i=0;i<9;i++)v=v*10+s[i];
    return v%hashsize;
}
 
int try_to_insert(int s){
    int h=hash(st[s]);
    int u=head[h];
    while(u){
        if(memcmp(st[u],st[s],sizeof(st[s]))==0)return 0;
        u=next[u];
    }
    next[s]=head[h];
    head[h]=s;
    return 1;
}
```

### 树链剖分结合线段树

```C++
#include <bits/stdc++.h>
using namespace std;

#define Temp template<typename T>
typedef long long LL;
#define mid ((l + r) >> 1)
#define Rint register int
const int mxn = 1e5 + 10;
Temp inline void read(T &x) {
    x = 0;T w = 1;char ch = getchar();
    while(!isdigit(ch) && ch!= '-')ch = getchar();
    if(ch == '-') w = -1, ch = getchar();
    while(isdigit(ch)) x = (x << 3) + (x << 1) + (ch^'0'), ch = getchar();
    x = x * w;
}
int n, m, r, mod, tot, cnt;
struct zex{int nex, to;}e[mxn << 1];
int head[mxn], w[mxn], wt[mxn], id[mxn], fa[mxn], son[mxn], siz[mxn], dep[mxn], top[mxn];

inline void add(int x, int y) {
    e[++ tot].to = y;e[tot].nex = head[x];head[x] = tot;
    e[++ tot].to = x;e[tot].nex = head[y];head[y] = tot;
}

inline void dfs1(int x, int fath) {
    fa[x] = fath, son[x] = 0;
    dep[x] = dep[fath] + 1, siz[x] = 1;
    for(int i = head[x]; i; i = e[i].nex) {
        int to = e[i].to;
        if(to == fath) continue;
        dfs1(to, x);
        siz[x] += siz[to];
        if(siz[to] > siz[son[x]]) son[x] = to;
    }
}

inline void dfs2(int x, int tops) {
    id[x] = ++ cnt; wt[cnt] = w[x];
    top[x] = tops;
    if(son[x])dfs2(son[x], tops);
    for(int i = head[x]; i; i = e[i].nex) {
        int to = e[i].to;
        if(to == fa[x] || to == son[x]) continue;
        dfs2(to, to);
    }
}

/*-----------------------------SegmentTree----------------------------------*/
int res[mxn << 2], laz[mxn << 2];

inline void pushup(int k){res[k] = (res[k << 1] + res[k << 1 | 1]) % mod;}

inline void pushdown(int k, int l, int r) {
    (res[k << 1] += laz[k] * (mid - l + 1) % mod) %= mod;
    (res[k << 1 | 1] += laz[k] * (r - mid) % mod) %= mod;
    (laz[k << 1] += laz[k]) %= mod, (laz[k << 1 | 1] += laz[k]) %= mod;
    laz[k] = 0;
}

inline void build(int k, int l, int r) {
    if(l == r) {res[k] = wt[l] % mod; return;}
    build(k << 1, l, mid);
    build(k << 1 | 1, mid + 1, r); 
    pushup(k);
}


inline void update(int k, int l, int r, int x, int y, int z) {
    if(y < l || x > r) return;
    if(x <= l && r <= y) {
        (res[k] += (r - l + 1) * z % mod) %= mod;
        (laz[k] += z) % mod;
        return ;
    }
    pushdown(k, l, r);
    update(k << 1, l, mid, x, y, z);
    update(k << 1 | 1, mid + 1, r, x, y, z);
    pushup(k);
}

inline int query(int k, int l, int r, int x, int y) {
    if(y < l || x > r) return 0;
    if(x <= l && r <= y) {return res[k];}
    pushdown(k, l, r);
    return (query(k << 1, l, mid, x, y) + query(k << 1 | 1, mid + 1, r, x, y)) % mod;
}

/*------------------------------end------------------------------------*/

inline void updRange(int x, int y, int z) {
    while(top[x] != top[y]) {
        if(dep[top[x]] < dep[top[y]]) swap(x, y);
        update(1, 1, n, id[top[x]], id[x], z);
        x = fa[top[x]];
    }
    if(dep[x] < dep[y]) swap(x, y);
    update(1, 1, n, id[y], id[x], z);
}

inline int qryRange(int x, int y) {
    int ans = 0;
    while(top[x] != top[y]) {
        if(dep[top[x]] < dep[top[y]]) swap(x, y);
        (ans += query(1, 1, n, id[top[x]], id[x])) %= mod;
        x = fa[top[x]];
    }
    if(dep[x] < dep[y]) swap(x, y);
    (ans += query(1, 1, n, id[y], id[x])) %= mod;
    return ans;
}

inline void updSon(int x, int k) {
    update(1, 1, n, id[x], id[x] + siz[x] - 1, k);
}

inline int qrySon(int x) {
    return query(1, 1, n, id[x], id[x] + siz[x] - 1);
}

int main() {
    read(n), read(m), read(r), read(mod);
    for(Rint i = 1; i <= n; i ++) {
        read(w[i]);
    }
    for(Rint i = 1; i <= n - 1; i ++) {
        int u, v; read(u), read(v);
        add(u, v);
    }
    dfs1(r, 0); dfs2(r, r);
    build(1, 1, n);
    for(Rint i = 1; i <= m; i ++) {
        int op, x, y, z; read(op);
        if(op == 1) {
            read(x), read(y), read(z);
            updRange(x, y, z);
        }else if(op == 2) {
            read(x), read(y);
            printf("%d\n", qryRange(x, y));
        }else if(op == 3) {
            read(x), read(z);
            updSon(x, z);
        }else {
            read(x);
            printf("%d\n", qrySon(x));
        }
    }
    return 0;
}


```

## 根号分治

```c++
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
const int mxn = 2e5 + 10;
const int mm = 500;
ll tag[mm + 10][mm + 10], a[mxn];
int main() {
    int n, q;
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    cin >> n >> q;
    for(int i = 1; i <= q; i ++) {
        int ty, x, y, d; cin >> ty;
        if(ty == 1) {
            cin >> x >> y >> d;
            if(x >= mm) {
                for(int i = y; i <= n; i += x) {
                    a[i] += d;
                }
            } else {
                tag[x][y] += d;
            }
        } else {
            cin >> x;
            ll t = a[x];
            for(int i = 1; i <= mm; i ++) {
                t += tag[i][x % i];
            }
            cout << t << '\n';
        }
    }
}
```

### 点分治

```c++
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

const int N = 101000;
int n, sz[N], maxs[N], L;
vector<pair<int, int>> e[N];
bool del[N];
ll ans;

void solve(int u, int s) {
	int ms = s + 1, root = -1;
	function<void(int, int)> dfs1 = [&](int u, int par) {
		sz[u] = 1;
		maxs[u] = 0;
		for (auto [v, w]: e[u]) {
			if (del[v] || v == par) continue;
			dfs1(v, u);
			sz[u] += sz[v];
			maxs[u] = max(maxs[u], sz[v]);
		}
		maxs[u] = max(maxs[u], s - sz[u]);
		if (maxs[u] < ms) ms = maxs[u], root = u;
	};
	dfs1(u, -1);
	vector<int> d1, d2;
	d1.push_back(0);
	auto calc = [&](vector<int> &d) {
		sort(d.begin(), d.end());
		int m = d.size();
		int r = m - 1;
		ll ans = 0;
		for (int i = 0; i < m; i++) {
			while (r >= 0 && d[i] + d[r] > L) --r;
			if (i < r) ans += r - i;
		}
		return ans;
	};
	for (auto [v, w] : e[root]) {
		if (del[v]) continue;
		d2.clear();
		function<void(int, int, int)> dfs2 = [&](int u, int par, int dep) {
			sz[u] = 1;
			d1.push_back(dep);
			d2.push_back(dep);
			for (auto [v, w] : e[u]) {
				if (del[v] || v == par) continue;
				sz[u] += sz[v];
				dfs2(v, u, dep + w);
			}
		};
		dfs2(v, root, w);
		ans -= calc(d2);
	}
	ans += calc(d1);
	del[root] = true;
	for (auto [v, w] : e[root]) {
		if (!del[v]) solve(v, sz[v]);
	}
}

int main() {
	scanf("%d", &n);
	for (int i = 1; i < n; i++) {
		int u, v, w;
		scanf("%d%d%d", &u, &v, &w);
		e[u].push_back(make_pair(v, w));
		e[v].push_back(make_pair(u, w));
	}
	scanf("%d", &L);
	solve(1, n);
	printf("%lld\n", ans);
}
```



## 字符串

### KMP

```C++
void kmp(char *s) {
	nxt[0] = 0;
	for(int i = 2; i <= n; i ++) {
		nxt[i] = nxt[i - 1];
		while(nxt[i] && s[i] != s[nxt[i] + 1]) nxt[i] = nxt[nxt[i]];
		nxt[i] += (s[i] == s[nxt[i] + 1]);
	}

}//预处理

void kmp(char *s, char *s1) { //模式匹配

	int j = 0;
	for(int i = 1; i <= n; i ++) {
		while(j && s[i] != s1[j + 1]) j = nxt[j];
		j += (s[i] == s1[j + 1]);
		if(j == n2) {printf("%d\n", i - n2 + 1); j = nxt[j];}
	}
}
```

### HASH

```C++
#define mp make_pair
#define fi first
#define se second
using ll = long long;
typedef pair<int, int> hashv;
const ll mod1 = 1e9 + 7;
const ll mod2 = 1e9 + 9;

hashv base = mp(13331, 2333);
hashv operator + (hashv a, hashv b) {
    int c1 = a.fi + b.fi, c2 = a.se + b.se;
    if(c1 >= mod1) c1 -= mod1;
    if(c2 >= mod2) c2 -= mod2;
    return mp(c1, c2);
}

hashv operator - (hashv a, hashv b) {
    int c1 = a.fi - b.fi, c2 = a.se - b.se;
    if(c1 < 0) c1 += mod1;
    if(c2 < 0) c2 += mod2;
    return mp(c1,c2);
}

hashv operator * (hashv a, hashv b) {
    return mp(1ll*a.fi*b.fi%mod1, 1ll*a.se*b.se%mod2);
}
```



### Manacher

```c++
//    O(1)判断回文 now[i]~now[j]   (p[(i+j)/2]>=(j-i)/2+1);
//
const int maxl = 11000050;
ll p[2 * maxl + 5];   //p[i]-1表示以i为中点的回文串长度
ll Manacher(string s)
{
    string now;
    ll len = s.size();
    for (ll i = 0; i < len; i++){      //将原串处理成%a%b%c%形式，保证长度为奇数
        now += '%';
        now += s[i];
    }
    now += '%';
    len = now.size();
    ll pos = 0, R = 0;
    for (ll i = 0; i < len; i++){
        if (i < R)
            p[i] = min(p[2 * pos - i], R - i);
        else
            p[i] = 1;
        while (i - p[i] >= 0 && i + p[i] < len && now[i - p[i]] == now[i + p[i]])
            p[i]++;
        if (i + p[i] > R){
            pos = i;
            R = i + p[i];
        }
    }
    ll MAX = 0;
    for (ll i = 0; i < len; i++){
        cout << i << " : " << p[i] - 1 << endl;            //p[i]-1为now串中以i为中点的回文半径，即是s中最长回文串的长度
        cout << now.substr(i - p[i] + 1, 2 * p[i] - 1) << endl;
        MAX = max(MAX, p[i] - 1);
    }
    return MAX;           //最长回文子串长度
}

```

### 找子串

```c++
vector<int> to[30];

bool check() {
    int len = strlen(t + 1);
    int pos = 0;
    for(int i = 1; i <= len; i ++) {
        int v = t[i] - 'a';
        pos = upper_bound(to[v].begin(), to[v].end(), pos) - to[v].begin();
        if(pos == to[v].size()) return false;
        pos = to[v][pos];
    } return true;
}
int main() {
    scanf("%d%d", &n, &q);
    scanf("%s", s + 1);
    for(int i = 1; i <= n; i ++)
        to[s[i] - 'a'].push_back(i);
    for(int i = 1; i <= q; i ++) {
        scanf("%s", t + 1);
        if(check()) printf("YES\n");
        else printf("NO\n");
    }
}
```

### ACAM

```C++
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

const int M = 26, N = 210000;
struct node {
	node *son[M], *go[M], *fail;
	int cnt;
}pool[N], *cur = pool, *d[N], *q[N], *root;

node *newnode() {
	return cur++;
}
int t, n;
char tt[N], s[N * 10];

void build() {
	t = 0;
	q[t++] = root;
	for (int i = 0; i < t; i++) {
		node *u = q[i];
		for (int j = 0; j < M; j++) {
			if (u->son[j]) {
				u->go[j] = u->son[j];
				if (u != root) u->go[j]->fail = u->fail->go[j];
				else u->go[j]->fail = root;
				q[t++] = u->son[j];
			}
			else {
				if (u != root) u->go[j] = u->fail->go[j];
				else u->go[j] = root;
			}
		}
	}	
}

int main() {
	scanf("%d", &n);
	root = newnode();
	for (int i = 0; i < n; i++) {
		scanf("%s", tt);
		int m = strlen(tt);
		node *p = root;
		for (int j = 0; j < m; j++) {
			int w = tt[j] - 'a';
			if (!p->son[w]) p->son[w] = newnode();
			p = p->son[w];
		}
		d[i] = p;
	}
	build();
	scanf("%s", s);
	node *p = root;
	int l = strlen(s);
	for (int i = 0; i < l; i++) {
		p = p->go[s[i] - 'a'];
		p->cnt++;
	}
	for (int i = t - 1; i; i--) {
		q[i]->fail->cnt += q[i]->cnt;
	}
	for (int i = 0; i < n; i++) {
		printf("%d\n", d[i]->cnt);
	}
}
```

### SA

```c++
const int N = 1e6 + 10;
char s[N];
int rk[N], sa[N], n;
// ht[i] = LCP(sa[i], sa[i - 1])
// LCP(sa[i], sa[j]) = min(ht[i] ~ ht[j])
//本质不同字串个数：n*(n + 1) / 2 - sum(ht);
//可重叠最长重复子串： max(ht);
//不可重叠：二分长度L，将相邻两项ht大于等于L的分组，找到每组中最早出现和最晚出现的，判断是否重叠
//可重叠重复k次子串，二分长度L，将相邻两项ht大于等于L的分组，是否有一组大于k
//多串用分隔符链接，二分答案，两个串可以用并查集维护所在组的信息。
//平方串：枚举长度L，分段，每段求LCP和LCS，能连起来就是平方串，有多少个平方串也可以顺便求出来
void buildSA(char *s, int *sa, int *rk, int *ht, int n, int m = 128) {
	static int x[N], y[N], c[N];
	s[n] = 0;
	for (int i = 0; i < m; i++) c[i] = 0;
	for (int i = 0; i < n; i++) c[x[i] = s[i]]++;
	for (int i = 1; i < m; i++) c[i] += c[i - 1];
	for (int i = n - 1; i >= 0; i--) sa[--c[x[i]]] = i;
	for (int k = 1; k < n; k <<= 1) {
		int p=0;
		for (int i = n - 1; i >= n - k; i--) y[p++] = i;
		for (int i = 0; i < n; i++) if (sa[i] >= k) y[p++] = sa[i] - k;
		for (int i = 0; i < m; i++) c[i] = 0;
		for (int i = 0; i < n; i++) c[x[y[i]]]++;
		for (int i = 1; i < m; i++) c[i] += c[i - 1];
		for (int i = n - 1; i >= 0; i--) sa[--c[x[y[i]]]] = y[i];
		swap(x, y);
		p = 1; x[sa[0]] = 0; y[n] = -1;
		for (int i = 1; i < n; i++) {
			if (y[sa[i - 1]] == y[sa[i]] && y[sa[i - 1] + k] == y[sa[i] + k])
				x[sa[i]] = p - 1;
			else
				x[sa[i]] = p++;
		}
		if (p == n) break;
		m = p;
	}
	for (int i = 0; i < n; i++) rk[sa[i]] = i;
	int k = 0;
	for (int i = 0; i < n; i++) {
		k = max(k - 1, 0);
		if (rk[i] == 0) continue;
		int j = sa[rk[i] - 1];
		while (s[i + k] == s[j + k]) k++;
		ht[rk[i]] = k;
	}
}
```

### SA(tarjen)

```c++
const int N = 1e6 + 10;
struct Suffix{
	int ht[N],rk[N],sa[N],y[N],c[N];
    int n,m;
    char s[N];
	void init(){
        n=strlen(s+1);
        m=300;
		for(int i=0;i<=m;i++) c[i]=0;
		for(int i=0;i<=2*n;i++) y[i]=0;
		for(int i=1;i<=n;i++) c[rk[i]=s[i]]++;
		for(int i=1;i<=m;i++) c[i]+=c[i-1];
		for(int i=n;i>=1;i--) sa[c[rk[i]]--]=i;
		for(int k=1;k<=n;k<<=1){
			int p=0;
			for(int i=n-k+1;i<=n;i++) y[++p]=i;
			for(int i=1;i<=n;i++){
				if(sa[i]>k){
					y[++p]=sa[i]-k;
				}
			} 
			for(int i=0;i<=m;i++) c[i]=0;
			for(int i=1;i<=n;i++) c[rk[i]]++;
			for(int i=1;i<=m;i++) c[i]+=c[i-1];
			for(int i=n;i>=1;i--) sa[c[rk[y[i]]]--]=y[i];
			for(int i=0;i<=n;i++) swap(rk[i],y[i]);
			rk[sa[1]]=p=1;
			for(int i=2;i<=n;i++){
				rk[sa[i]]=(y[sa[i]] == y[sa[i-1]] && y[sa[i]+k] == y[sa[i-1]+k] ? p : ++p);
			}
			if(p>=n) break;
			m=p;
		}
		for(int i=1,k=0;i<=n;i++){
			if(k)k--;
			int j=sa[rk[i]-1];
			while(s[i+k] == s[j+k] )k++;
			ht[rk[i]] = k;
		}
	}
    void writ()
    {
        printf("%s\n",s+1);
        for(int i=1;i<=n;i++)cout<<sa[i]<<" ";;cout<<"\n";
        for(int i=1;i<=n;i++)cout<<ht[i]<<" ";;cout<<"\n";
        for(int i=1;i<=n;i++)cout<<rk[i]<<" ";;cout<<"\n";
    }
    
};
Suffix suf;

```



## DP

### 最长上升子序列的nlogn算法（含二分查找）

```c++
for(int j = l + 1; j < r; j ++)
            if (a[j] >= a[l] && a[j] <= a[r]) // a[j]满足条件，可以出现在区间中
			{
				auto pos = upper_bound(LIS.begin(), LIS.end(), a[j]);
				if (pos == LIS.end()) LIS.push_back(a[j]);
				else *pos = a[j];
			}
		ans += (r-l-1) - LIS.size(); // 此段答案为段长-LIS长度
```

### DAG上的动态规划

```c++
void DP(){
    //memset(dp,0x3f,sizeof(dp));
    for(int i=1;i<=m;i++)dp[i][n]=mp[i][n];
    for(int j=n-1;j>=1;j--){
        for(int i=1;i<=m;i++){
            for(int k=-1;k<=1;k++){
                int kk=i+k;
                if(kk==0)kk=m;
                if(kk==m+1)kk=1;
                dp[i][j]=min(dp[i][j],dp[kk][j+1]+mp[i][j]);
            }
        }
    }
}
```

### 单调队列优化

```c++
#include <bits/stdc++.h>
using namespace std;
const int N = 1e4 + 10;
int n, m;
int dp[N], c[N][2];
int main() {
    scanf("%d%d", &n, &m);
    for(int i = 1; i <= n; i ++) {
        int v, w, t; scanf("%d%d%d", &v, &w, &t);
        for(int j = 0; j < v; j ++) {
            int k = 0, l = 1;
            for(int p = j, x = 1; p <= m; p += v, x ++) {
                int e = dp[p] - x * w, r = x + t;
                for(; k >= l && c[k][0] <= e; k --);
                c[++k][0] = e, c[k][1] = r;
                dp[p] = c[l][0] + x * w;
                for(; l <= k && c[l][1] <= x; l ++);
            }
        }
    }
    printf("%d", dp[m]);
}
```



### 数位dp模板

```c++
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

ll dp[20][2][10][5];

ll dfs(int rem, int exist, int last, int inc) {
 // 还剩多少位，存不存在连续三位递增，末尾的数值，末尾的递增长度
	if (rem == 0) {
		return exist;
	}
	if (dp[rem][exist][last][inc] != -1) {
		return dp[rem][exist][last][inc];
	}
	ll &ans = dp[rem][exist][last][inc];
	ans = 0;
	for (int i = 0; i <= 9; i++) {
		int inc_ = (i > last)? min(inc + 1, 3) : 1;
		ans += dfs(rem - 1, exist | (inc_ == 3), i, inc_);
	}
	return ans;
}

ll solve(ll x) {  // 1 ... x
	x += 1; // 1 ... x - 1
	vector<int> d;
	while (x) {
		d.push_back(x % 10);
		x /= 10;
	}
	reverse(d.begin(), d.end());
	int m = d.size();
	// 前导0
	ll ans = 0;
	for (int i = 1; i <= m - 1; i++) {
		for (int j = 1; j <= 9; j++)
			ans += dfs(i - 1, 0, j, 1);
	}
	int exist = 0, last = 0, inc = 0;
	for (int i = 0; i < m; i++) {
		for (int j = (i == 0) ? 1 : 0; j < d[i]; j++) {
			int inc_ = (j > last)? min(inc + 1, 3) : 1;
			ans += dfs(m - i - 1, exist | (inc_ == 3), j, inc_);
		}
		inc = (d[i] > last)? min(inc + 1, 3) : 1;
		exist |= (inc == 3);
		last = d[i];
	}
	return ans;
}

int main() {
	ll l, r;
	scanf("%lld%lld", &l, &r);
	memset(dp, -1 ,sizeof(dp));
	printf("%lld\n", solve(r) - solve(l - 1));
}
```

### 高位前缀和

```c++
for(int i = 0; i < M; i ++) {
		for(int j = 0; j < (1 << M); j ++)
			if(j & (1 << i)) 
				f[j] += f[j - (1 << i)];
	}
```

## 博弈



```c++
int sg[1010], n = 100;
int calcsg(int x) {
    if(x == 0 || x % 4 == 1 || x % 4 == 2) return x;
    if(x % 4 == 3) return x + 1;
    else return x - 1;
}

int main() {
    for(int i = 1; i <= n; i ++) {
        set<int> s;
        for(int j = 0; j < i; j ++)
            s.insert(sg[i]);
        for(int j = 1; j < i; j ++)
            s.insert(sg[j] ^ sg[i - j]);
        while(!s.count(sg[i])) sg[i] ++;
        printf("%d %d\n", i, sg[i]);
        assert(sg[i] == calcsg(i));
    }
}
```



## 宏

```cpp
//keeping_running
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native")
#include <bits/stdc++.h>
using namespace std;
#define pb push_back
#define MT make_tuple
#define MP make_pair
#define pii pair<int,int>
#define pdd pair<double,double>
#define fi first
#define se second
const int N = 2e5 +10;
#define MOD 1000000007
#define PI (acos(-1.0))
#define EPS 1e-6
#define MMT(s,a) memset(s, a, sizeof s)
#define GO(i,a,b) for(int i = (a); i < (b); ++i)
#define GOE(i,a,b) for(int i = (a); i <= (b); ++i)
#define OG(i,a,b) for(int i = (a); i > (b); --i)
#define OGE(i,a,b) for(int i = (a); i >= (b); --i)

typedef unsigned long long ull;
typedef long long ll;
typedef double db;
typedef long double ldb;
typedef stringstream sstm;
int fx[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,-1},{-1,1}};

template<typename T> using maxHeap = priority_queue<T, vector<T>, less<T> >;
template<typename T> using minHeap = priority_queue<T, vector<T>, greater<T> >;

inline char nc(){ static char buf[1000000], *p1 = buf, *p2 = buf; return p1 == p2 && (p2 = (p1 = buf) + fread(buf,1,1000000,stdin),p1 == p2) ? EOF : *p1++; }
#define nc getchar
template<typename T> inline int read(T& sum){ char ch = nc(); if(ch == EOF || ch == -1) return 0; int tf = 0; sum = 0; while((ch < '0' || ch > '9') && (ch != '-')) ch = nc(); tf = ((ch == '-') && (ch = nc())); while(ch >= '0' && ch <= '9') sum = sum*10 + (ch-48), ch = nc(); (tf) && (sum = -sum); return 1; }
template<typename T,typename... Arg> inline int read(T& sum,Arg&... args){ int ret = read(sum); if(!ret) return 0; return read(args...); }
template<typename T1,typename T2> inline void read(T1* a,T2 num){ for(int i = 0; i < num; i++){read(a[i]);} }
template<typename T1,typename T2> inline void read(T1* a,T2 bn,T2 ed){ for(;bn <= ed; bn++){read(a[bn]);} }
inline void read(char* s){ char ch = nc(); int num = 0; while(ch != ' ' && ch != '\n' && ch != '\r' && ch != EOF){s[num++] = ch;ch = nc();} s[num] = '\0'; }
inline void read(string& s){ static char tp[1000005]; char ch = nc(); int num = 0; while(ch != ' ' && ch != '\n' && ch != '\r' && ch != EOF){tp[num++] = ch;ch = nc();} tp[num] = '\0';    s = (string)tp; }
template<typename T> inline void print(T k){ int num = 0,ch[20]; if(k == 0){ putchar('0'); return ; } (k<0)&&(putchar('-'),k = -k); while(k>0) ch[++num] = k%10, k /= 10; while(num) putchar(ch[num--]+48); }
template<typename T,typename... Arg> inline void print(T k,Arg... args){ print(k),putchar(' '); print(args...);}
template<typename T1,typename T2> inline void print(T1* a,T2 num){ print(a[0]); for(int i = 1; i < num; i++){putchar(' '),print(a[i]);} }
template<typename T1,typename T2> inline void print(T1* a,T2 bn,T2 ed){ print(a[bn++]); for(;bn <= ed; bn++){putchar(' '),print(a[bn]);} }
/*math*/
template<typename T> inline T gcd(T a, T b){ return b==0 ? a : gcd(b,a%b); }
template<typename T> inline T lowbit(T x){ return x&(-x); }
template<typename T> inline bool mishu(T x){ return x>0?(x&(x-1))==0:false; }
template<typename T1,typename T2, typename T3> inline ll q_mul(T1 a,T2 b,T3 p){ ll w = 0; while(b){ if(b&1) w = (w+a)%p; b>>=1; a = (a+a)%p; } return w; }
template<typename T,typename T2> inline ll f_mul(T a,T b,T2 p){ return (a*b - (ll)((long double)a/p*b)*p+p)%p; }
template<typename T1,typename T2, typename T3> inline ll q_pow(T1 a,T2 b,T3 p){ ll w = 1; while(b){ if(b&1) w = (w*a)%p; b>>=1; a = (a*a)%p;} return w; }
template<typename T1,typename T2, typename T3> inline ll s_pow(T1 a,T2 b,T3 p){ ll w = 1; while(b){ if(b&1) w = q_mul(w,a,p); b>>=1; a = q_mul(a,a,p);} return w; }
template<typename T> inline ll ex_gcd(T a, T b, T& x, T& y){ if(b == 0){ x = 1, y = 0; return (ll)a; } ll r = exgcd(b,a%b,y,x); y -= a/b*x; return r;/*gcd*/ }
template<typename T1,typename T2> inline ll com(T1 m, T2 n) { int k = 1;ll ans = 1; while(k <= n){ ans=((m-k+1)*ans)/k;k++;} return ans; }
template<typename T> inline bool isprime(T n){ if(n <= 3) return n>1; if(n%6 != 1 && n%6 != 5) return 0; T n_s = floor(sqrt((db)(n))); for(int i = 5; i <= n_s; i += 6){ if(n%i == 0 || n%(i+2) == 0) return 0; } return 1; }
/*data structure*/
template<class T> struct BIT {T c[N];int size;void init(int s) {size = s;for (int i = 1; i <= size; i++) c[i] = 0;}T query(int x) {assert(x <= size);T s = 0;for (; x; x -= x & (-x)) {s += c[x];}return s;}void modify(int x, T s) {assert(x != 0);for (; x <= size; x += x & (-x)) {c[x] += s;}}};
/* ----------------------------------------------------------------------------------------------------------------------------------------------------------------- */


```

### 宏

```c++
#include <bits/stdc++.h>
using namespace std;
#define rep(i, a, n) for(int i=a;i<n;i ++)
#define per(i, a, n) for(int i=n-1;i>=a;i--)
#define pb push_back
#define eb emplace_back
#define mp make_pair
#define all(x) (x).begin, (x).end()
#define fi first
#define se second 
#define SZ(x) ((int) (x).size())
typedef vector<int> VI;
typedef basic_string<int> BI;
typedef long long ll;
typedef pair<int, int> PII;
typedef double db;
mt19937 mrand(random_device{}());
const ll mod = 1e9 + 7;
int rnd(int x){return mrand() % x;}
ll qpow(ll a, ll b) {ll res=1;for(a%=mod;b;b>>=1){if(b&1)res=res*a%mod;a=a*a%mod;}return res;}
ll gcd(ll a, ll b){return b?gcd(b,a%b):a;}
//head

```

### midint

````c++
template<int MOD, int RT> struct mint {
	static const int mod = MOD;
	static constexpr mint rt() { return RT; } // primitive root for FFT
	int v; explicit operator int() const { return v; } // explicit -> don't silently convert to int
	mint():v(0) {}
	mint(ll _v) { v = int((-MOD < _v && _v < MOD) ? _v : _v % MOD);
		if (v < 0) v += MOD; }
	bool operator==(const mint& o) const {
		return v == o.v; }
	friend bool operator!=(const mint& a, const mint& b) { 
		return !(a == b); }
	friend bool operator<(const mint& a, const mint& b) { 
		return a.v < b.v; }
   
	mint& operator+=(const mint& o) { 
		if ((v += o.v) >= MOD) v -= MOD; 
		return *this; }
	mint& operator-=(const mint& o) { 
		if ((v -= o.v) < 0) v += MOD; 
		return *this; }
	mint& operator*=(const mint& o) { 
		v = int((ll)v*o.v%MOD); return *this; }
	mint& operator/=(const mint& o) { return (*this) *= inv(o); }
	friend mint pow(mint a, ll p) {
		mint ans = 1; assert(p >= 0);
		for (; p; p /= 2, a *= a) if (p&1) ans *= a;
		return ans; }
	friend mint inv(const mint& a) { assert(a.v != 0); 
		return pow(a,MOD-2); }
		
	mint operator-() const { return mint(-v); }
	mint& operator++() { return *this += 1; }
	mint& operator--() { return *this -= 1; }
	friend mint operator+(mint a, const mint& b) { return a += b; }
	friend mint operator-(mint a, const mint& b) { return a -= b; }
	friend mint operator*(mint a, const mint& b) { return a *= b; }
	friend mint operator/(mint a, const mint& b) { return a /= b; }
};
````

### simp

```c++
const int MOD=998244353; 
using mi = mint<MOD,5>; // 5 is primitive root for both common mods

namespace simp {
	vector<mi> fac,ifac,invn;
	void check(int x) {
		if (fac.empty()) {
			fac={mi(1),mi(1)};
			ifac={mi(1),mi(1)};
			invn={mi(0),mi(1)};
		}
		while (SZ(fac)<=x) {
			int n=SZ(fac),m=SZ(fac)*2;
			fac.resize(m);
			ifac.resize(m);
			invn.resize(m);
			for (int i=n;i<m;i++) {
				fac[i]=fac[i-1]*mi(i);
				invn[i]=mi(MOD-MOD/i)*invn[MOD%i];
				ifac[i]=ifac[i-1]*invn[i];
			}
		}
	}
	mi gfac(int x) {
		check(x); return fac[x];
	}
	mi ginv(int x) {
		check(x); return invn[x];
	}
	mi gifac(int x) {
		check(x); return ifac[x];
	}
	mi binom(int n,int m) {
		if (m < 0 || m > n) return mi(0);
		return gfac(n)*gifac(m)*gifac(n - m);
	}
}

```



## stl

```c++
iota(p.begin(), p.end(), 0);//生成上升序列
sort(a.begin(), a.end(), greater());//降序排列
priority_queue<int, vector<int>, greater<int>> p
freopen("in.txt", "r", stdin);
freopen("out.txt", "w", stdout);//w
alls.erase(unique(alls.begin(), alls.end()), alls.end());//离散化
__builtin_ctz(w)// 二进制下末尾0的个数
__builtin_ffs(x)//返回 x 的最后一位 1 是从后向前第几位
__builtin_clz(x)//返回 x 二进制下前导 0 的个数
__builtin_popcount(x)//返回 x 二进制下 1 的个数
__builtin_parity(x)//返回 x 的 1 的个数的奇偶性    
```

## 小trick

```markdown
数据范围           最多因子个数                   最多因子个数对应的数
1e4                    64                               7560
1e5                    128                              83160
1e6                    240                              720720
1e7                    448                              8648640
1e8                    768                              73513440
1e9                    1344                             735134400
```

对拍

```bat
g++ data.cpp -o data -g
g++ std.cpp -o std -g
g++ bf.cpp -o bf -g
:loop
    data.exe>1.txt
    std.exe<1.txt>2.txt
    bf.exe<1.txt>3.txt
    fc 2.txt 3.txt
if not errorlevel 1 goto loop
pause
goto loop
```

卡时

```c++
struct gettime{
    clock_t star,ends;
    void begin(){
        star = clock();
    }
    void end(){
        ends = clock();
        cout <<"Running Time : "<<(double)(ends - star)/ CLOCKS_PER_SEC << endl;
    }
} tim;
int main()
{
    tim.begin();
    tim.end();
    return 0;

}
```

### c++对拍

```c++
#include <cstdio>
#include <cstdlib>
#include <ctime>
using namespace std;
int main()
{
    while (1) //一直循环，直到找到不一样的数据
    {
        system("data.exe > in.txt");
        system("baoli.exe < in.txt > baoli.txt");
        system("std.exe < in.txt > std.txt");
        if (system("fc std.txt baoli.txt")) //当 fc 返回1时，说明这时数据不一样
            break;                          //不一样就跳出循环
    }
    return 0;
}

```

### 随机数生成

```c++
std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
```

