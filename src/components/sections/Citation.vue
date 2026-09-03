<script>
import { CopyDocument } from '@element-plus/icons-vue'

export default {
  components: { CopyDocument },
  data() {
    return {
      bibtex: [
        "@inproceedings{wu2024dept,",
        "    title={DePT: Decoupled Prompt Tuning},",
        "    author={Wu, Shihan and Gao, Lianli and Shen, Heng Tao and Song, Jingkuan},",
        "    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},",
        "    year={2024}",
        "}",
      ],    }
  },
  methods: {
    copyVal() {
      let oInput = document.createElement('textarea');
      let text = document.getElementById('bibtex').innerText;
      oInput.value = text;
      document.body.appendChild(oInput);
      oInput.select();
      document.execCommand('Copy');
      this.$message.success('Copy Successfully');
      oInput.remove();
    }
  }
}

</script>

<template>
  <div id="citation" v-animate-onscroll="'animated fadeInUp'">
      
      <el-row justify="center">
        <h1 class="section-title">BibTeX</h1>
      </el-row>
      
      <el-row justify="center">
        <el-col class='bibtex' :xs="28" :sm="24" :md="20" :lg="16" :xl="12" @click="copyVal()" >
          <div class="bibtex-head">
            <span class="copy-tip">Click anywhere to copy BibTeX</span>
            <span class="copy-btn">
              <el-icon :size="14"><CopyDocument /></el-icon>
              Copy
            </span>
          </div>
          <el-row>
              <el-scrollbar style="margin: 0px 20px 5px 20px;">
                <pre id="bibtex"><code v-for="b in bibtex">{{ b }}<br/></code></pre>
              </el-scrollbar>
          </el-row>
        </el-col>
      </el-row>

  </div>
</template>

<style scoped>

/* 卡片属性 */
.bibtex {
  margin: 20px 0px;
  padding-top: 5px;
  background: #ffffff;
  border: 1px solid var(--pcd-border);
  border-radius: 14px;
  cursor: copy;
  transition: box-shadow 0.3s ease, transform 0.3s ease;
}

/* 卡片悬浮 */
.bibtex:hover {
  box-shadow: var(--pcd-shadow-lift);
  transform: translateY(-2px);
}

/* 卡片点击 */
.bibtex:active{
  box-shadow: var(--pcd-shadow-soft);
}

/* 卡片头部：提示 + 复制按钮 */
.bibtex-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px 0;
}

.copy-tip {
  color: var(--pcd-muted);
  font-family: 'Google Sans', sans-serif;
  font-size: 14px;
  transition: color 0.2s ease;
}

.bibtex:hover .copy-tip {
  color: var(--pcd-accent);
}

.copy-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 14px;
  border-radius: 999px;
  background: rgba(50, 115, 220, 0.08);
  color: var(--pcd-accent);
  font-family: 'Google Sans', sans-serif;
  font-size: 13px;
  font-weight: 600;
  transition: background 0.2s ease, color 0.2s ease;
}

.bibtex:hover .copy-btn {
  background: var(--pcd-accent);
  color: #ffffff;
}

pre {
  border: none;
  border-radius: 0px;
  padding: 10px;
  background: none;
}

pre code {
  font-size: 18px;
  background: none;
}

</style>