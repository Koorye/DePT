<script setup>
import motivation from '../mds/motivation.mdx';
</script>

<template>
  <div v-animate-onscroll="'animated fadeInUp'">
    <el-row justify="center">
        <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="12">
          <h1>Motivation</h1>
            <div class="caption-box">
                <el-image src="./motivation.png" class="caption-img" fit="cover" />
                <motivation class="caption-text" />
            </div>
        </el-col>
    </el-row>

  </div>
</template>

<style scoped>

.caption-box {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 20px 0;
    gap: 40px;
}

.caption-img {
  width: 150%;
  max-width: 400px;
  border-radius: 12px;
  border: 1px solid var(--pcd-border);
  box-shadow: var(--pcd-shadow-soft);
}

.caption-text {
    margin-left: 10px;
}

</style>
