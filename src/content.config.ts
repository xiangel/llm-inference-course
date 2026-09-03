import { defineCollection, z } from "astro:content";
import { docsLoader } from "@astrojs/starlight/loaders";
import { docsSchema } from "@astrojs/starlight/schema";

export const collections = {
  docs: defineCollection({
    loader: docsLoader(),
    schema: docsSchema({
      extend: z.object({
        tags: z.array(z.string()).optional(),
        difficulty: z
          .enum(["beginner", "intermediate", "advanced"])
          .optional(),
        section: z.string().optional(),
        readingTime: z.number().optional(),
      }),
    }),
  }),
};
