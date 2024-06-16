import { promises as fs } from 'fs'
import path from 'path'
import matter from 'gray-matter';

const __filename = new URL(import.meta.url).pathname;
const __dirname = path.dirname(__filename);


async function generate() {
  const posts = await fs.readdir(path.join(__dirname, '..', 'pages', 'posts'))
  const allTags = {}
  await Promise.all(
    posts.map(async (name) => {
      if (name.startsWith('index.')) return

      const content = await fs.readFile(
        path.join(__dirname, '..', 'pages', 'posts', name)
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
