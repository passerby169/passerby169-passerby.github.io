;(function () {
  const { createApp, computed, onMounted, onBeforeUnmount } = Vue
  const { createPinia, defineStore, storeToRefs } = Pinia

  const docsContent = window.__scDocs || {}

  const docPages = [
    {
      id: 'guide',
      title: 'Guide',
      summary: '搭建环境、加载数据并完成第一个干预模拟。',
      path: 'guide/index.html',
      badge: 'Guide',
      group: 'Guide'
    },
    {
      id: 'api',
      title: 'API',
      summary: '类型、actions 与 devtools 接入说明。',
      path: 'api/index.html',
      badge: 'API',
      group: 'API'
    },
    {
      id: 'cookbook',
      title: 'Cookbook',
      summary: '迁移、日志、可视化等常用配方集合。',
      path: 'cookbook/index.html',
      badge: 'Cookbook',
      group: 'Cookbook'
    }
  ]

  const navLinks = [
    { label: 'Guide', path: 'guide/index.html' },
    { label: 'API', path: 'api/index.html' },
    { label: 'Cookbook', path: 'cookbook/index.html' },
    {
      label: 'Links',
      children: [
        { label: 'GitHub', href: 'https://github.com/', external: true },
        { label: 'Issues', href: 'https://github.com/issues', external: true }
      ]
    }
  ]

  const hero = {
    eyebrow: 'Causal Single-cell Store',
    title: ['scCAFM', 'Intuitive stores for experiments'],
    description:
      'Type safe, modular, and explainable by design. Use Pinia + Vue to narrate interventions powered by scCAFM.',
    primary: { label: 'Get Started', path: 'guide/index.html' },
    secondary: { label: 'See Cookbook', path: 'cookbook/index.html' }
  }

  const heroLinks = [
    { label: 'Watch intro video', href: 'https://www.bilibili.com', external: true },
    { label: 'Get cheatsheet', path: 'guide/index.html' }
  ]

  const featureCards = [
    {
      title: 'Intuitive',
      description: 'Stores feel like components, orchestrating assays with minimal APIs.',
      icon: '✨'
    },
    {
      title: 'Type Safe',
      description: 'Strong typings for perturbations keep IDEs fully assisted.',
      icon: '🧬'
    },
    {
      title: 'Devtools ready',
      description: 'Replay CAFM events via Pinia Devtools for reproducible audits.',
      icon: '🛠️'
    },
    {
      title: 'Modular',
      description: 'Compose stores per cohort; bundlers split them automatically.',
      icon: '🧩'
    },
    {
      title: 'SSR friendly',
      description: 'Hydrate once and serve docs with SEO using the same store.',
      icon: '🚀'
    },
    {
      title: 'Featherweight',
      description: 'Pinia core ≈1.5 kb so dashboards stay responsive.',
      icon: '🪶'
    }
  ]

  const useSiteStore = defineStore('site', {
    state: () => ({
      theme: 'dark',
      searchOpen: false,
      searchQuery: '',
      navLinks,
      hero,
      heroLinks,
      featureCards,
      docPages,
      activeDocId: null,
      docHtml: '',
      docToc: [],
      docLoading: false
    }),
    getters: {
      docNav(state) {
        return ['Guide', 'API', 'Cookbook'].map((groupTitle) => ({
          title: groupTitle,
          pages: state.docPages.filter((page) => page.group === groupTitle)
        }))
      },
      docMeta(state) {
        return state.docPages.find((page) => page.id === state.activeDocId)
      },
      searchResults(state) {
        const query = state.searchQuery.trim().toLowerCase()
        if (!query) return []
        return state.docPages
          .filter(
            (page) =>
              page.title.toLowerCase().includes(query) || page.summary.toLowerCase().includes(query)
          )
          .map((page) => ({ ...page }))
          .slice(0, 5)
      }
    },
    actions: {
      toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark'
        document.body.setAttribute('data-theme', this.theme)
      },
      hydrateTheme() {
        document.body.setAttribute('data-theme', this.theme)
      },
      openSearch() {
        this.searchOpen = true
        queueMicrotask(() => {
          const input = document.querySelector('[data-search-input]')
          input && input.focus()
        })
      },
      closeSearch() {
        this.searchOpen = false
        this.searchQuery = ''
      },
      resolveLink(path) {
        if (!path || /^https?:/.test(path)) return path;
        const base = document.body?.dataset?.root || '.';
        // 移除 base 中可能的多余斜杠
        const trimmedBase = base.replace(/\/+$/, '');
        // 移除路径中可能的前导斜杠
        const cleanedPath = path.replace(/^\/+/, '');
        // 处理根路径为 . 或空的情况
        if (!trimmedBase || trimmedBase === '.') {
          return cleanedPath ? `./${cleanedPath}` : '.';
        }
        // 拼接基础路径和目标路径
        return `${trimmedBase}/${cleanedPath}`;
      },
      navigateTo(path) {
        if (!path) return
        window.location.href = this.resolveLink(path)
      },
      // 在 site.js 中找到 setDoc 方法（约 162-185 行），替换 Slugger 相关逻辑
// site.js 中更新 setDoc 方法
// site.js 中更新 setDoc 方法（约 163-196 行）
setDoc(id, forceRefresh = false) {
  // 如果是同一文档且不强制刷新，直接返回
  if (!forceRefresh && this.activeDocId === id) return;

  if (!id) return;
  this.activeDocId = id;
  this.searchOpen && this.closeSearch();
  this.docLoading = true;

  // 动态加载Markdown文件
  const loadMarkdown = async () => {
    try {
      // 使用 data-docs 属性作为基础路径（已在 HTML 中修正为 ./docs）
      const docsPath = document.body?.dataset?.docs || './docs';
      const timestamp = forceRefresh ? `?t=${Date.now()}` : '';
      // 使用 resolveLink 方法处理路径
      const url = this.resolveLink(`${docsPath}/${id}.md${timestamp}`);
      
      const response = await fetch(url);
      if (!response.ok) throw new Error(`文件不存在: ${url}`);
      return await response.text();
    } catch (e) {
      console.error('加载Markdown失败:', e);
      return `# 内容加载失败\n\n无法加载文档 "${id}". 错误: ${e.message}`;
    }
  };

  // 处理加载后的Markdown内容
  loadMarkdown().then(markdown => {
    const marked = window.marked;
    if (marked) {
      const tokens = marked.lexer(markdown);
      // 自定义标题ID生成函数
      const slugify = (text) => {
        return text.toLowerCase()
          .replace(/[^a-z0-9]+/g, '-')
          .replace(/^-+|-+$/g, '');
      };
      // 生成目录
      this.docToc = tokens
        .filter((token) => token.type === 'heading' && token.depth <= 3)
        .map((token) => ({
          id: slugify(token.text),
          title: token.text,
          depth: token.depth
        }));
      // 解析为HTML
      this.docHtml = marked.parser(tokens, {
        headerIds: true,
        mangle: false,
        slugger: { slug: slugify }
      });
    } else {
      this.docHtml = `<pre>${markdown}</pre>`;
      this.docToc = [];
    }
    this.docLoading = false;
  });
}
    }
  })

  function mountLanding(selector) {
    const pinia = createPinia()
    const app = createApp({
      setup() {
        const store = useSiteStore()
        const { theme, navLinks, hero, heroLinks, featureCards, docPages, searchOpen, searchQuery, searchResults } =
          storeToRefs(store)

        const onKeyDown = (event) => {
          if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
            event.preventDefault()
            store.openSearch()
          } else if (event.key === 'Escape' && store.searchOpen) {
            store.closeSearch()
          }
        }

        onMounted(() => {
          store.hydrateTheme()
          window.addEventListener('keydown', onKeyDown)
        })

        onBeforeUnmount(() => {
          window.removeEventListener('keydown', onKeyDown)
        })

        return {
          theme,
          navLinks,
          hero,
          heroLinks,
          featureCards,
          docPages,
          searchOpen,
          searchQuery,
          searchResults,
          toggleTheme: store.toggleTheme,
          openSearch: store.openSearch,
          closeSearch: store.closeSearch,
          resolveLink: store.resolveLink,
          navigateTo: store.navigateTo
        }
      }
    })

    app.use(pinia)
    app.mount(selector)
  }

  function mountDocPage(selector, docId) {
    const pinia = createPinia()
    const app = createApp({
      setup() {
        const store = useSiteStore()
        const { theme, navLinks, docPages, docNav, docHtml, docToc, docLoading, docMeta, searchOpen, searchQuery, searchResults } =
          storeToRefs(store)
  
        const onKeyDown = (event) => {
          if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
            event.preventDefault()
            store.openSearch()
          } else if (event.key === 'Escape' && store.searchOpen) {
            store.closeSearch()
          }
        }
  
        onMounted(() => {
          store.hydrateTheme()
          store.setDoc(docId)
          window.addEventListener('keydown', onKeyDown)
          
          // 添加WebSocket监听
          if (window.WebSocket) {
            // 连接到文件更新WebSocket服务
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(`${wsProtocol}//${window.location.host}/file-watcher`);
            
            ws.onopen = () => {
              console.log('Connected to file watcher');
              // 订阅Markdown文件变化
              ws.send(JSON.stringify({
                type: 'subscribe',
                pattern: '*.md'
              }));
            };
            
            ws.onmessage = (event) => {
              try {
                const data = JSON.parse(event.data);
                if (data.type === 'fileChanged') {
                  // 提取文件名作为文档ID
                  const fileName = data.path.split('/').pop();
                  const updatedDocId = fileName.replace('.md', '');
                  
                  // 如果是当前查看的文档，刷新内容
                  if (store.activeDocId === updatedDocId) {
                    store.setDoc(updatedDocId, true);
                  }
                }
              } catch (e) {
                console.error('Error processing websocket message:', e);
              }
            };
            
            ws.onerror = (error) => {
              console.error('WebSocket error:', error);
            };
            
            ws.onclose = () => {
              console.log('File watcher connection closed. Reconnecting...');
              // 3秒后尝试重连
              setTimeout(() => {
                // 简单重连逻辑
              }, 3000);
            };
          }
        })
  
        onBeforeUnmount(() => {
          window.removeEventListener('keydown', onKeyDown)
        })
  
        return {
          theme,
          navLinks,
          docPages,
          docNav,
          docHtml,
          docToc,
          docLoading,
          docMeta,
          searchOpen,
          searchQuery,
          searchResults,
          setDoc: store.setDoc,
          toggleTheme: store.toggleTheme,
          openSearch: store.openSearch,
          closeSearch: store.closeSearch,
          resolveLink: store.resolveLink
        }
      }
    })
  
    app.use(pinia)
    app.mount(selector)
  }

  window.scCAFMFront = {
    mountLanding,
    mountDocPage
  }
})()
