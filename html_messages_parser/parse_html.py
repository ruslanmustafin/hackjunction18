from lxml import html
import csv

num_messages = 216

with open('result.csv', "w") as f:
    writer = csv.writer(f)
    for i_n in range(1, num_messages + 1):
        txt = []
        times = []
        from_names = []
        with open('data/messages{}.html'.format(i_n), encoding='utf-8') as html_file:
            tree = html.fromstring(html_file.read())
            path = '//div[@class="history"]/div[@class="message default clearfix" or @class="message default clearfix joined"]/div[@class="body"]'
            messages = tree.xpath(path)

            prev_name = ''
            for m in messages:
                children = m.getchildren()
                if len(children) == 3:
                    if children[0].get('class') == 'pull_right date details' and \
                            children[2].get('class') == 'text':
                        times.append(children[0].get('title'))
                        txt.append(children[2].text.strip())
                        if children[1].get('class') == 'from_name':
                            from_names.append(children[1].text.strip())
                            prev_name = children[1].text.strip()
                        else:
                            from_names.append(prev_name)
                elif len(children) == 2:
                    if children[0].get('class') == 'pull_right date details' and \
                            children[1].get('class') == 'text':
                        times.append(children[0].get('title'))
                        txt.append(children[1].text.strip())
                        from_names.append(prev_name)
                elif len(children) == 4:
                    if children[2].get('class') == 'reply_to details' or \
                            children[2].get('class') == 'reply_to details':
                        times.append(children[0].get('title'))
                        from_names.append(children[1].text.strip())
                        txt.append(children[3].text.strip())
                        prev_name = children[1].text.strip()

            print(len(txt))
            print(len(times))
            print(len(from_names))

        rows = zip(from_names, times, txt)
        for row in rows:
            writer.writerow(row)
