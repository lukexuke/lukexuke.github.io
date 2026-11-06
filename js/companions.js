(() => {
  const root = document.documentElement;
  const button = document.querySelector('.companion-toggle');
  if (!button) return;
  const label = button.querySelector('[data-companion-label]');
  const status = document.getElementById('companion-status');
  const decorations = [...document.querySelectorAll('.companion-decoration')];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const timers = new Map();
  const flights = new Set();
  let enabled = false;
  let generation = 0;
  let observer;
  let buttonTimer;
  let effectLayer;
  const namespace = 'http://www.w3.org/2000/svg';

  const visible = rect => rect.bottom > 76 && rect.top < window.innerHeight - 10 && rect.right > 0 && rect.left < window.innerWidth;
  const svgElement = (name, attributes = {}) => {
    const element = document.createElementNS(namespace, name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  };
  const clearFlights = () => {
    flights.forEach(flight => {
      window.cancelAnimationFrame(flight.frame);
      window.clearTimeout(flight.timer);
      flight.onCancel?.();
      flight.node.remove();
    });
    flights.clear();
  };
  const layer = () => {
    if (!document.createElementNS || !window.requestAnimationFrame) return null;
    if (!effectLayer) {
      effectLayer = svgElement('svg', {'class':'companion-effects','aria-hidden':'true','focusable':'false'});
      const definitions = svgElement('defs');
      const filter = svgElement('filter', {id:'companion-energy-glow',x:'-70%',y:'-70%',width:'240%',height:'240%'});
      filter.appendChild(svgElement('feGaussianBlur',{stdDeviation:2.6,result:'glow'}));
      const merge = svgElement('feMerge');
      merge.appendChild(svgElement('feMergeNode',{in:'glow'}));
      merge.appendChild(svgElement('feMergeNode',{in:'SourceGraphic'}));
      filter.appendChild(merge);definitions.appendChild(filter);effectLayer.appendChild(definitions);
      document.body.appendChild(effectLayer);
    }
    effectLayer.setAttribute('viewBox', `0 0 ${window.innerWidth} ${window.innerHeight}`);
    effectLayer.setAttribute('width',window.innerWidth);
    effectLayer.setAttribute('height',window.innerHeight);
    return effectLayer;
  };

  // The raised hand in the full-body Gekko artwork, after its transparent margins are cropped.
  const gekkoHand = () => {
    const character = document.getElementById('companion-gekko');
    if (!character) return null;
    const rect = character.getBoundingClientRect();
    return {x:rect.left+rect.width*.41,y:rect.top+rect.height*.15};
  };
  const pointOnArc = (start,control,end,t) => ({
    x:(1-t)*(1-t)*start.x+2*(1-t)*t*control.x+t*t*end.x,
    y:(1-t)*(1-t)*start.y+2*(1-t)*t*control.y+t*t*end.y
  });
  const playDizzyFlight = (decoration, reverse, version, rect, interruptedPosition) => {
    const canvas=layer(), hand=gekkoHand();
    if (!canvas || !hand) return false;
    // Launch while Gekko is still stepping in, without waiting for a scroll event.
    const launchAt=window.performance.now()+(reverse?0:160);
    const duration=reverse?720:960;
    const size=Math.max(76,Math.min(112,rect.width));
    const group=svgElement('g',{'class':'companion-dizzy-flight'});
    const actor=svgElement('g');
    const bubble=svgElement('circle',{r:size*.25,fill:'#78d7ff55',stroke:'#b8f4ff','stroke-width':2.5,filter:'url(#companion-energy-glow)',opacity:0});
    actor.appendChild(bubble);
    const creature=svgElement('g',{'class':'companion-dizzy-actor'});
    const cell=svgElement('svg',{x:-size/2,y:-size/2,width:size,height:size,viewBox:'1 0 1 1',overflow:'hidden'});
    cell.appendChild(svgElement('image',{href:'./img/games/gekko-companions.png',width:2,height:1,preserveAspectRatio:'none'}));
    creature.appendChild(cell);actor.appendChild(creature);group.appendChild(actor);
    canvas.appendChild(group);group.style.opacity='0';
    decoration.removeAttribute('data-companion-motion');
    decoration.setAttribute('data-companion-in-flight','');
    const flight={node:group,frame:0,timer:0,decoration,point:null,onCancel:()=>decoration.removeAttribute('data-companion-in-flight')};
    flights.add(flight);
    let geometry;
    const tick=time => {
      if(generation!==version || reducedMotion.matches)return;
      if(time<launchAt){flight.frame=window.requestAnimationFrame(tick);return;}
      const scrollX=window.scrollX||0, scrollY=window.scrollY||0;
      if(!geometry){
        const handNow=gekkoHand();
        const landing=decoration.getBoundingClientRect();
        // If Gekko is above the viewport, the throw enters from the same column at the top.
        const pickup={x:handNow.x+scrollX,y:Math.max(-size,Math.min(window.innerHeight+size,handNow.y))+scrollY};
        const perch={x:landing.left+landing.width/2+scrollX,y:landing.top+landing.height/2+scrollY};
        const start=reverse?(interruptedPosition||perch):pickup;
        const end=reverse?pickup:perch;
        const control={x:(start.x+end.x)/2,y:Math.min(start.y,end.y)-Math.min(320,Math.max(180,Math.abs(end.y-start.y)*.5+100))};
        geometry={start,end,control};
      }
      // Follow responsive layout changes while keeping the launch point fixed in the document.
      const targetRect=decoration.getBoundingClientRect();
      const currentHand=gekkoHand();
      geometry.end=reverse
        ? {x:currentHand.x+scrollX,y:Math.max(-size,Math.min(window.innerHeight+size,currentHand.y))+scrollY}
        : {x:targetRect.left+targetRect.width/2+scrollX,y:targetRect.top+targetRect.height/2+scrollY};
      geometry.control={x:(geometry.start.x+geometry.end.x)/2,y:Math.min(geometry.start.y,geometry.end.y)-Math.min(320,Math.max(180,Math.abs(geometry.end.y-geometry.start.y)*.5+100))};
      const progress=Math.min(1,(time-launchAt)/duration);
      const travel=reverse?1-Math.pow(1-progress,2):progress;
      const point=pointOnArc(geometry.start,geometry.control,geometry.end,travel);
      const scale=reverse?(interruptedPosition?.scale||1)*(1-.74*travel):.24+.76*Math.min(1,travel*1.9);
      const rotation=reverse?Math.sin(travel*Math.PI)*14:-Math.sin(travel*Math.PI)*22;
      flight.point={x:point.x,y:point.y,scale};
      group.style.opacity='1';
      actor.setAttribute('transform',`translate(${point.x-scrollX} ${point.y-scrollY}) rotate(${rotation}) scale(${scale})`);
      if(reverse){bubble.setAttribute('opacity',Math.min(1,travel*2));creature.style.opacity=String(Math.max(0,1-travel*1.7));}
      if(progress<1){flight.frame=window.requestAnimationFrame(tick);return;}
      actor.remove();decoration.removeAttribute('data-companion-in-flight');
      if(!reverse && enabled){
        decoration.style.setProperty('--companion-delay','0ms');
        decoration.setAttribute('data-companion-motion','summon');
        timers.set(decoration,window.setTimeout(()=>{
          if(generation===version)decoration.removeAttribute('data-companion-motion');
          timers.delete(decoration);
        },1900));
      }
      const end=geometry.end;
      const impact=svgElement('circle',{cx:end.x-scrollX,cy:end.y-scrollY,r:10,fill:'none',stroke:'#8de9ff','stroke-width':3,filter:'url(#companion-energy-glow)'});
      group.appendChild(impact);impact.animate?.([{r:'10px',opacity:1},{r:reverse?'25px':'42px',opacity:0}],{duration:480,fill:'both',easing:'ease-out'});
      group.style.transition='opacity 480ms ease-out';group.style.opacity='0';
      flight.timer=window.setTimeout(()=>{group.remove();flights.delete(flight);},520);
    };
    flight.frame=window.requestAnimationFrame(tick);
    return true;
  };

  const stopScene = () => {
    observer?.disconnect();observer = null;
    window.clearTimeout(buttonTimer);
    timers.forEach(timer => window.clearTimeout(timer));timers.clear();
    clearFlights();root.removeAttribute('data-companion-motion');
    decorations.forEach(decoration => {
      decoration.removeAttribute('data-companion-motion');
      decoration.removeAttribute('data-companion-pending');
      decoration.removeAttribute('data-companion-in-flight');
      decoration.removeAttribute('data-companion-hold');
    });
  };
  const animateDecoration = (decoration, reverse, version, delay = 0, interruptedPosition) => {
    const rect = decoration.getBoundingClientRect();
    decoration.style.setProperty('--companion-delay', `${delay}ms`);
    decoration.removeAttribute('data-companion-pending');
    if(decoration.classList.contains('companion-dizzy') && playDizzyFlight(decoration,reverse,version,rect,interruptedPosition))return;
    decoration.setAttribute('data-companion-motion',reverse?'recall':'summon');
    timers.set(decoration, window.setTimeout(() => {
      if (generation===version) decoration.removeAttribute('data-companion-motion');
      timers.delete(decoration);
    },reverse?1100:2800+delay));
  };
  const setCompanions = (next, animate = true) => {
    const previouslyShown = decorations.filter(decoration => !decoration.hasAttribute('data-companion-pending'));
    const interruptedFlight=[...flights].find(flight=>flight.decoration?.classList.contains('companion-dizzy'));
    const interruptedPosition=interruptedFlight?.point;
    generation++;stopScene();enabled=next;
    const version=generation;
    root.setAttribute('data-companions',enabled?'on':'off');
    button.setAttribute('aria-pressed',String(enabled));
    const action=enabled?'收回丢丢':'召唤丢丢';
    button.setAttribute('aria-label',action);button.setAttribute('title',action);
    if(label)label.textContent=action;
    if(status && animate)status.textContent=enabled?'丢丢已出发，伙伴们到栏目标题上啦。':'丢丢和伙伴们已收回。';
    if (!animate || reducedMotion.matches) return;
    root.setAttribute('data-companion-motion',enabled?'summon':'recall');
    buttonTimer=window.setTimeout(() => {if(generation===version)root.removeAttribute('data-companion-motion');},enabled?2800:1600);
    if (!enabled) {
      const returningDizzy=previouslyShown.find(decoration=>decoration.classList.contains('companion-dizzy') && (!interruptedFlight || interruptedPosition));
      previouslyShown.forEach(decoration => {
        if(!visible(decoration.getBoundingClientRect()) && !decoration.classList.contains('companion-dizzy'))return;
        if(decoration.classList.contains('companion-dizzy') && interruptedFlight && !interruptedPosition)return;
        if(decoration.classList.contains('portrait-character') && returningDizzy){
          // Keep Gekko's hand present until the blue pickup orb reaches it.
          decoration.setAttribute('data-companion-hold','');
          timers.set(decoration,window.setTimeout(()=>{
            if(generation!==version)return;
            timers.delete(decoration);decoration.removeAttribute('data-companion-hold');
            animateDecoration(decoration,true,version);
          },840));
        }else animateDecoration(decoration,true,version,0,interruptedPosition);
      });
      return;
    }
    if ('IntersectionObserver' in window) {
      observer=new window.IntersectionObserver(entries => {
        if(!enabled || generation!==version || reducedMotion.matches || !observer)return;
        entries.forEach(entry => {
          if(entry.isIntersecting && entry.target.hasAttribute('data-companion-pending')) {
            observer.unobserve(entry.target);animateDecoration(entry.target,false,version);
          }
        });
      },{rootMargin:'-75px 0px -10px 0px',threshold:.05});
    }
    decorations.forEach((decoration,index) => {
      if(decoration.classList.contains('companion-dizzy') || !observer || visible(decoration.getBoundingClientRect()))animateDecoration(decoration,false,version,index*70);
      else {decoration.setAttribute('data-companion-pending','');observer.observe(decoration);}
    });
  };

  // Always start clean, independently of theme and localStorage.
  setCompanions(false,false);
  button.addEventListener('click',() => setCompanions(!enabled));
  reducedMotion.addEventListener('change',() => {if(reducedMotion.matches)stopScene();});
  const syncViewport=() => {if(effectLayer)layer();};
  window.addEventListener('resize',syncViewport,{passive:true});
  window.visualViewport?.addEventListener('resize',syncViewport,{passive:true});
  window.addEventListener('pagehide',stopScene);
})();
