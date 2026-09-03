<template>
  <footer class="site-footer">
    <div class="footer-line"></div>
    <div class="footer-content">
      <span>© 2024 DePT · Decoupled Prompt Tuning</span>
      <span class="footer-divider">|</span>
      <a href="https://koorye.github.io/home/" target="_self">Homepage</a>
      <span class="footer-divider">|</span>
      <a href="https://github.com/Koorye/DePT/" target="_blank">GitHub</a>
      <span class="footer-divider">|</span>
      <a href="https://arxiv.org/abs/2309.07439" target="_blank">arXiv</a>
    </div>
  </footer>
</template>

<style scoped>

.site-footer {
  margin-top: 64px;
  padding: 28px 16px 36px;
  text-align: center;
}

.footer-line {
  width: min(720px, 80%);
  height: 1px;
  margin: 0 auto 20px;
  background: linear-gradient(90deg, transparent, var(--pcd-border), transparent);
}

.footer-content {
  font-family: 'Inter', 'Noto Sans', sans-serif;
  font-size: 14px;
  color: var(--pcd-muted);
  display: flex;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}

.footer-divider {
  color: var(--pcd-border);
}

.footer-content a {
  color: var(--pcd-muted);
  transition: color 0.2s ease;
}

.footer-content a:hover {
  color: var(--pcd-accent);
}

</style>
