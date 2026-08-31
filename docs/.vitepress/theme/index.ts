import DefaultTheme from "vitepress/theme";
import type { Theme } from "vitepress";
import KvCalculator from "./components/KvCalculator.vue";
import SamplingLab from "./components/SamplingLab.vue";
import RooflineLab from "./components/RooflineLab.vue";
import Quiz from "./components/Quiz.vue";
import Checklist from "./components/Checklist.vue";
import Roadmap from "./components/Roadmap.vue";
import DecoderDiagram from "./components/DecoderDiagram.vue";
import PrefillDecodeDiagram from "./components/PrefillDecodeDiagram.vue";
import TransformerFlow from "./components/TransformerFlow.vue";
import KvCacheFlow from "./components/KvCacheFlow.vue";
import ComingSoon from "./components/ComingSoon.vue";
import "./custom.css";

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component("KvCalculator", KvCalculator);
    app.component("SamplingLab", SamplingLab);
    app.component("RooflineLab", RooflineLab);
    app.component("Quiz", Quiz);
    app.component("Checklist", Checklist);
    app.component("Roadmap", Roadmap);
    app.component("DecoderDiagram", DecoderDiagram);
    app.component("PrefillDecodeDiagram", PrefillDecodeDiagram);
    app.component("TransformerFlow", TransformerFlow);
    app.component("KvCacheFlow", KvCacheFlow);
    app.component("ComingSoon", ComingSoon);
  },
} satisfies Theme;
