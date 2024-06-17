import { promises as fs } from 'fs'
import path from 'path'
import matter from 'gray-matter';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


async function generate() {
  const postsDir = path.resolve(__dirname, '..', 'pages', 'posts');
  const posts = await fs.readdir(postsDir)
  const allTags = {}
  await Promise.all(
    posts.map(async (name) => {
      if (name.startsWith('index.')) return

      const content = await fs.readFile(
        path.join(postsDir, name)
      )
      const frontmatter = matter(content)

      if (frontmatter.data.draft) return
      frontmatter.data.tag.split(',').forEach((tag) => {
        tag = tag.trim()
        allTags[tag] = allTags[tag] ? allTags[tag] + 1 : 1
      })
    })
  )

  await fs.writeFile('./public/tags.json', JSON.stringify(allTags))
}

generate()
