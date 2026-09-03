<script>
import Title from './sections/Title.vue'
import Abstract from './sections/Abstract.vue'
import Motivation from './sections/Motivation.vue'
import Method from './sections/Method.vue'
import Experiments from './sections/Experiments.vue'
import Citation from './sections/Citation.vue'
import Highlights from './sections/Highlights.vue'
import Poster from './sections/Poster.vue'
import Footer from './sections/Footer.vue'

export default {
  components: {
    Title,
    Abstract,
    Motivation,
    Method,
    Experiments,
    Citation,
    Highlights,
    Poster,
    Footer,
  }
}
</script>

<template>
  <Title/>
  <Highlights/>
  <Poster/>
  <Abstract/>
  <Motivation/>
  <Method/>
  <Experiments/>
  <Citation/>
  <Footer/>
</template>
