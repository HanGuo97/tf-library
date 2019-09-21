// Install Node.js without sudo: https://gist.github.com/isaacs/579814
const puppeteer = require('puppeteer');
const inquirer = require('inquirer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.goto('https://kubem.its.unc.edu:32002');
  await page.click("input[value='Request Token']");

  var username_question = [{
    type: 'password',
    name: 'username',
    message: "Username: ?",
  }]

  var password_question = [{
    type: 'password',
    name: 'password',
    message: "Password:",
  }]

  await inquirer.prompt(username_question).then(async (answers) => {
    // console.log(`Hi ${answers['username']}!`)
    await page.type('input[name="login"]', answers['username']);
    await inquirer.prompt(password_question).then(async (answers) => {
      // console.log(`Hi ${answers['password']}!`)
      await page.type('input[name="password"]', answers['password']);
    })
  })

  await page.click("button#submit-login");

  
  await page.waitForSelector(".hljs.yaml");
  let html = await page.content();
  html = await html.split('<code class="hljs yaml">')[1].split('</code></pre>')[0].trim()
  await console.log(html);
  await browser.close();
})();
